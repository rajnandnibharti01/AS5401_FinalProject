import csv
import sys
import time
import warnings
from functools import lru_cache
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

# 1. CONFIGURATION
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
QUICK = "--quick" in sys.argv
DATA_DIR = ARGS[0] if ARGS else "."

CASES = {                      # id: (delta [mm], T [C], mu)
    1: (0.0008, -120, 0.542), 2: (0.0005, -120, 0.542), 3: (0.0001, -120, 0.542),
    4: (0.0008,   25, 0.600), 5: (0.0005,   25, 0.600), 6: (0.0001,   25, 0.600),
    7: (0.0008,  200, 0.670), 8: (0.0005,  200, 0.670), 9: (0.0001,  200, 0.670),
}
TRAIN = [5, 4, 6, 2, 8]
TEST = [1, 3, 7, 9]
BASELINE = 5

INPUT_MODE = "delta"           # "delta": input = indentation depth; "force": input = plateau F_n
V_SLIDE = 1.0                  # mm/s
RAMP_FRAC = 0.10               # smooth-step ramp = first 10 % of the sliding step
PLATEAU = (0.15, 0.90)         # fraction of step used for steady-state values
K_WEAR = 1e-9                  # placeholder Archard coefficient (cancels in relative errors)
FAR_MARGIN = 0.5               # mm; nodes farther than this from the track = "far field"

QUAD_MULT = 10.0               # quadratic ridge weight = QUAD_MULT * linear weight
Q_BLOWUP = 50.0                # |scaled q| above this = unstable rollout
LONG_FACTOR = 3                # selection guard: rollout must stay bounded for 3x the horizon
R_LIST = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30]
LAMS = 10.0 ** np.arange(-8, 5)
SAME_R = [3, 5, 8]
N_SWAP = 15
SEED = 0
EXTRAP_FRAMES = 71             # time-extrapolation: train on frames 0..70, predict 71..100
ABAQUS_WALL_S = {}             # optional {case_id: seconds}; enables the speed-up report

if QUICK:                      # fast smoke test
    R_LIST, LAMS, SAME_R, N_SWAP = [3, 8], 10.0 ** np.arange(-4, 3, 2), [3], 3

VARIANTS = {                   # ablation ladder (each step changes one ingredient)
    "DMDc":        dict(quad=False, const=False, gate=False),   # linear baseline
    "DMDc+c":      dict(quad=False, const=True,  gate=False),   # + constant term
    "DMDc-gated":  dict(quad=False, const=False, gate=True),    # + inputs gated by speed ramp
    "OpInf":       dict(quad=True,  const=True,  gate=False),   # original quadratic OpInf
    "OpInf-gated": dict(quad=True,  const=False, gate=True),    # quadratic + gating
}
INPUT_NAMES = ["d-input", "T-input", "ramp"]


# 2. LOADING, INPUTS, CHECKS, AUDIT
def trap(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def smoothstep(xi):
    xi = np.clip(xi, 0.0, 1.0)
    return xi ** 3 * (10 - 15 * xi + 6 * xi ** 2)


def pmask(t):
    return (t >= PLATEAU[0] * t[-1]) & (t <= PLATEAU[1] * t[-1])


def load_case(cid):
    d = np.load(f"{DATA_DIR}/case{cid:02d}.npz")
    delta, Tc, mu = CASES[cid]
    X, t = d["X"], d["t"]
    RF1, RF2 = d["RF1"], d["RF2"]
    return dict(id=cid, X=X, Xt=X - X[:, [0]], x0=X[:, 0], t=t, xy=d["xy"], P=d["P"], S=d["S"],
                RF1=RF1, RF2=RF2, U1=d["U1rp"], delta=delta, T=Tc, mu=mu,
                cof=np.abs(RF1) / np.maximum(np.abs(RF2), 1e-12),
                Fn_pl=float(np.abs(RF2)[pmask(t)].mean()))


def build_inputs(cases):
    """Inputs are centred on the baseline (deviation form) and scaled to O(1).
    u_raw  = [d, T, ramp]                (constant factors + ramp)
    u_gate = [d*ramp, T*ramp, ramp]      (nothing is forced until the ball moves)"""
    b = cases[BASELINE]
    if INPUT_MODE == "force":
        dev = max(abs(cases[i]["Fn_pl"] - b["Fn_pl"]) for i in TRAIN) or 1.0
    for c in cases.values():
        t = c["t"]
        amp = smoothstep(t / (RAMP_FRAC * t[-1]))
        ds = (c["delta"] - b["delta"]) / 0.0003 if INPUT_MODE == "delta" else (c["Fn_pl"] - b["Fn_pl"]) / dev
        Ts = (c["T"] - b["T"]) / 175.0
        c["amp"] = amp
        c["u_raw"] = np.vstack([np.full_like(t, ds), np.full_like(t, Ts), V_SLIDE * amp])
        c["u_gate"] = np.vstack([ds * amp, Ts * amp, V_SLIDE * amp])


def check_consistency(cases):
    ref = cases[TRAIN[0]]
    for cid, c in cases.items():
        assert c["X"].shape == ref["X"].shape, f"case {cid}: X shape differs"
        assert np.allclose(c["xy"], ref["xy"]), f"case {cid}: node coordinates/ordering differ"
        assert np.allclose(c["t"], ref["t"]), f"case {cid}: time stamps differ"
        assert not np.isnan(c["X"]).any(), f"case {cid}: NaN in X"
        exp = V_SLIDE * trap(c["amp"], c["t"])
        got = c["U1"][-1] - c["U1"][0]
        if abs(got - exp) > 0.02 * abs(exp):
            print(f"  WARNING case {cid}: RP travel {got:.4f} vs ramp integral {exp:.4f}")


def top_nodes(xy):
    top = np.where(np.isclose(xy[:, 1], xy[:, 1].max()))[0]
    return top[np.argsort(xy[top, 0])]


def data_audit(cases, top):
    print("\nDATA AUDIT (one line per case)")
    print(f"{'case':>4} {'delta_um':>8} {'T':>5} {'mu':>6} {'Fn_plateau':>11} {'pmax_MPa':>9} "
          f"{'nodes_mid':>9} {'COF/mu':>7} {'k*Fn':>9}")
    for cid in sorted(cases):
        c = cases[cid]
        pl = pmask(c["t"])
        mid = c["P"][top, len(c["t"]) // 2]
        print(f"{cid:>4} {1e3 * c['delta']:8.2f} {c['T']:5d} {c['mu']:6.3f} {c['Fn_pl']:11.2f} "
              f"{c['P'].max():9.0f} {int((mid > 0.01 * mid.max()).sum()):9d} "
              f"{c['cof'][pl].mean() / c['mu']:7.3f} {K_WEAR * c['Fn_pl']:9.2e}")
    for lo, hi in ((3, 1), (6, 4), (9, 7)):
        if lo in cases and hi in cases:
            ratio = cases[hi]["Fn_pl"] / cases[lo]["Fn_pl"]
            dr = cases[hi]["delta"] / cases[lo]["delta"]
            print(f"  F_n(high delta)/F_n(low delta) = {ratio:.2f} (delta ratio {dr:.1f}, "
                  f"elastic line contact would give about {dr:.0f})")


def combine(case_list):
    return dict(Xt=np.hstack([c["Xt"] for c in case_list]), U=np.hstack([c["u_raw"] for c in case_list]),
                case_id=np.concatenate([np.full(c["Xt"].shape[1], c["id"]) for c in case_list]),
                frame=np.concatenate([np.arange(c["Xt"].shape[1]) for c in case_list]),
                x0=np.stack([c["x0"] for c in case_list], axis=1))


# 3. POD, FEATURES, FITTING, ROLLOUT
def nfr(c, frames):
    return frames.get(c["id"], c["Xt"].shape[1])


def rel_err(A, B):
    if not np.all(np.isfinite(A)):
        return np.inf
    return float(np.linalg.norm(A - B) / max(np.linalg.norm(B), 1e-30))


def sf(v):
    return float(v) if np.isfinite(v) else np.inf


def get_u(c, gate):
    return c["u_gate"] if gate else c["u_raw"]


def make_basis(cases, frames, rmax):
    Z = np.hstack([c["Xt"][:, :nfr(c, frames)] for c in cases])
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    V = U[:, :min(rmax, U.shape[1])]
    sc = np.abs(V.T @ Z).max(axis=1)
    sc[sc == 0] = 1.0
    return dict(V=V, s=s, sc=sc)


def project(c, basis, r):
    return (basis["V"][:, :r].T @ c["Xt"]) / basis["sc"][:r, None]


@lru_cache(maxsize=None)
def _triu(r):
    return np.triu_indices(r)


def features(Q, U, spec):
    parts = []
    if spec["const"]:
        parts.append(np.ones((1, Q.shape[1])))
    parts += [Q, U]
    if spec["quad"]:
        iu = _triu(Q.shape[0])
        parts.append(Q[iu[0]] * Q[iu[1]])
    return np.vstack(parts)


def reg_vec(r, p, spec, lam):
    b = []
    if spec["const"]:
        b.append(np.zeros(1))
    b += [np.full(r, lam), np.full(p, lam)]
    if spec["quad"]:
        b.append(np.full(r * (r + 1) // 2, lam * QUAD_MULT))
    return np.concatenate(b)


def fit_rom(basis, cases, frames, r, lam, spec):
    Q0, Q1, UU = [], [], []
    for c in cases:                                  # pairs only inside a case
        nf = nfr(c, frames)
        q = project(c, basis, r)[:, :nf]
        Q0.append(q[:, :-1]); Q1.append(q[:, 1:]); UU.append(get_u(c, spec["gate"])[:, :nf - 1])
    Q0, Q1, UU = np.hstack(Q0), np.hstack(Q1), np.hstack(UU)
    D = features(Q0, UU, spec)
    reg = reg_vec(r, UU.shape[0], spec, lam) + 1e-12
    Op = np.linalg.solve(D @ D.T + np.diag(reg), D @ Q1.T).T
    return dict(Op=Op, r=r, p=UU.shape[0], spec=spec, lam=lam, D=D,
                onestep=float(np.linalg.norm(Op @ D - Q1) / np.linalg.norm(Q1)))


def rollout(model, U):
    Q = np.zeros((model["r"], U.shape[1]))           # x~_0 = 0  ->  q_0 = 0
    for k in range(U.shape[1] - 1):
        q = model["Op"] @ features(Q[:, [k]], U[:, [k]], model["spec"])[:, 0]
        if (not np.all(np.isfinite(q))) or np.abs(q).max() > Q_BLOWUP:
            Q[:, k + 1:] = np.nan
            break
        Q[:, k + 1] = q
    return Q


def predict_state(model, basis, c):
    r = model["r"]
    Q = rollout(model, get_u(c, model["spec"]["gate"]))
    return Q, basis["V"][:, :r] @ (Q * basis["sc"][:r, None])


def rho_lin(model):
    off = 1 if model["spec"]["const"] else 0
    r = model["r"]
    return float(np.abs(np.linalg.eigvals(model["Op"][:, off:off + r])).max())


def long_ok(model, c):
    """Selection guard: bounded rollout for LONG_FACTOR times the horizon (inputs held constant)."""
    U = get_u(c, model["spec"]["gate"])
    Ul = np.hstack([U, np.repeat(U[:, [-1]], (LONG_FACTOR - 1) * (U.shape[1] - 1), axis=1)])
    Q = rollout(model, Ul)
    return bool(np.all(np.isfinite(Q)))


def make_fold_bases(train, rmax):
    return {c["id"]: make_basis([t for t in train if t["id"] != c["id"]], {}, rmax)
            for c in train if c["id"] != BASELINE}


def cv_select(train, cases, spec, r_list, lams, fold_bases, basis_full):
    """Leave-one-case-out over the non-baseline training cases (baseline always kept).
    Candidates are ranked by mean held-out rollout error; the first one that also passes the
    long-rollout guard on all training cases is returned."""
    holds = [c["id"] for c in train if c["id"] != BASELINE]
    err = np.full((len(holds), len(r_list), len(lams)), np.inf)
    for a, hid in enumerate(holds):
        tr = [c for c in train if c["id"] != hid]
        b = fold_bases[hid]
        for i, r in enumerate(r_list):
            for j, lam in enumerate(lams):
                m = fit_rom(b, tr, {}, r, lam, spec)
                err[a, i, j] = rel_err(predict_state(m, b, cases[hid])[1], cases[hid]["Xt"])
    mean = err.mean(axis=0)
    order = np.dstack(np.unravel_index(np.argsort(mean, axis=None), mean.shape))[0]
    for i, j in order:
        if not np.isfinite(mean[i, j]):
            break
        m = fit_rom(basis_full, train, {}, r_list[i], lams[j], spec)
        if all(long_ok(m, c) for c in train):
            return r_list[i], lams[j], float(mean[i, j]), mean
    return r_list[0], lams[-1], np.inf, mean        


# 4. OUTPUT MAPS (COF, F_n, CPRESS) AND WEAR
def ridge(Z, Y, lam):
    return np.linalg.solve(Z @ Z.T + lam * np.eye(Z.shape[0]), Z @ Y.T).T


def make_pbasis(train, top, e_tol=1e-4, rp_max=10):
    Z = np.hstack([c["P"][top] for c in train])
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    en = np.cumsum(s ** 2) / np.sum(s ** 2)
    rp = int(min(rp_max, np.searchsorted(en, 1 - e_tol) + 1))
    V = U[:, :rp]
    sc = np.abs(V.T @ Z).max(axis=1)
    sc[sc == 0] = 1.0
    return dict(V=V, sc=sc, mask=(Z.max(axis=1) > 0), rp=rp)


def out_feats(Q, U, amp, with_const):
    poly = np.vstack([amp ** 2, amp ** 3, amp ** 4])       # lets COF saturate like stick -> slip
    parts = [Q, U, poly]
    if with_const:
        parts.append(np.ones((1, Q.shape[1])))
    return np.vstack(parts)


def fit_maps(basis, pb, cases_list, top, r, spec, lam=1e-3):
    Zc, Zf, Yc, Yf, A = [], [], [], [], []
    for c in cases_list:
        q = project(c, basis, r)
        U, amp = get_u(c, spec["gate"]), c["amp"][None, :]
        Zc.append(out_feats(q, U, amp, not spec["gate"]))   # gated COF has no intercept: COF(0) = 0
        Zf.append(out_feats(q, U, amp, True))
        Yc.append(c["cof"][None, :]); Yf.append(np.abs(c["RF2"])[None, :])
        A.append((pb["V"].T @ c["P"][top]) / pb["sc"][:, None])
    Zc, Zf = np.hstack(Zc), np.hstack(Zf)
    Yc, Yf, A = np.hstack(Yc), np.hstack(Yf), np.hstack(A)
    fsc = np.abs(Yf).max()
    return dict(Wc=ridge(Zc, Yc / max(Yc.max(), 1e-12), lam), csc=max(Yc.max(), 1e-12),
                Wf=ridge(Zf, Yf / fsc, lam), fsc=fsc, Wa=ridge(Zf, A, lam))


def predict_outputs(maps, pb, Q, c, spec):
    U, amp = get_u(c, spec["gate"]), c["amp"][None, :]
    cof = (maps["Wc"] @ out_feats(Q, U, amp, not spec["gate"]))[0] * maps["csc"]
    Zf = out_feats(Q, U, amp, True)
    Fn = np.abs((maps["Wf"] @ Zf)[0]) * maps["fsc"]
    P = np.clip(pb["V"] @ ((maps["Wa"] @ Zf) * pb["sc"][:, None]), 0, None) * pb["mask"][:, None]
    return cof, Fn, P


def wear(Ptop, U1, k=K_WEAR):
    """Archard: h_i = k * sum_j mean(p_i^j, p_i^(j+1)) * ds_j, ds from the imposed RP travel."""
    return k * (0.5 * (Ptop[:, :-1] + Ptop[:, 1:]) * np.diff(U1)).sum(axis=1)


def train_bundle(cases_list, spec, r, lam, top, pb, basis):
    model = fit_rom(basis, cases_list, {}, r, lam, spec)
    return dict(basis=basis, model=model, pb=pb, spec=spec,
                maps=fit_maps(basis, pb, cases_list, top, r, spec))


def baseline_regression(train, c, only_state=False, top=None):
    """Static parametric surrogate: every snapshot/output is regressed in time, frame by frame,
    on [1, d-input, T-input] over the training cases (a simple POD-RBF-style baseline)."""
    G = np.array([[1.0, t["u_raw"][0, 0], t["u_raw"][1, 0]] for t in train])
    g = np.array([1.0, c["u_raw"][0, 0], c["u_raw"][1, 0]])

    def reg(stack):
        flat = stack.reshape(len(train), -1)
        return (g @ np.linalg.lstsq(G, flat, rcond=None)[0]).reshape(stack.shape[1:])
    Xh = reg(np.stack([t["Xt"] for t in train]))
    if only_state:
        return Xh
    cof = reg(np.stack([t["cof"] for t in train]))
    Fn = reg(np.stack([np.abs(t["RF2"]) for t in train]))
    P = np.clip(reg(np.stack([t["P"][top] for t in train])), 0, None)
    return Xh, cof, Fn, P


# 5. METRICS
def rise_frames(cof, pl):
    ss = cof[pl].mean()
    idx = np.where(cof >= 0.9 * ss)[0]
    return float(idx[0]) if len(idx) else np.nan


def metrics(c, Xh, cof_h, Fn_h, P_h, top):
    Xt = c["Xt"]
    N = Xt.shape[0] // 2
    xs = c["xy"][:, 0]
    rows_s = np.r_[top, N + top]
    track = np.where(c["P"].max(axis=1) > 0)[0]
    rows_c = np.r_[track, N + track]
    xt = xs[top][c["P"][top].max(axis=1) > 0]
    far = np.where((xs < xt.min() - FAR_MARGIN) | (xs > xt.max() + FAR_MARGIN))[0]
    rows_f = np.r_[far, N + far]
    x = xs[top]; dx = np.gradient(x)
    t = c["t"]; pl = pmask(t)
    P_fe = c["P"][top]; Fn_fe = np.abs(c["RF2"])
    h_fe, h_ro = wear(P_fe, c["U1"]), wear(P_h, c["U1"])
    m = {"case": c["id"]}
    m["e_u"] = rel_err(Xh, Xt)
    m["e_surf"] = rel_err(Xh[rows_s], Xt[rows_s])
    m["e_contact"] = rel_err(Xh[rows_c], Xt[rows_c])
    m["e_far_share"] = sf(np.linalg.norm((Xh - Xt)[rows_f]) / np.linalg.norm(Xt))
    m["e_u_full"] = rel_err(Xh + c["x0"][:, None], c["X"])
    m["cof_rmse"] = sf(np.sqrt(np.mean((cof_h - c["cof"]) ** 2)))
    m["cof_rmse_trans"] = sf(np.sqrt(np.mean((cof_h[:15] - c["cof"][:15]) ** 2)))
    m["cof_ss_abs"] = sf(abs(cof_h[pl].mean() - c["cof"][pl].mean()))
    m["cof_over_mu"] = sf(cof_h[pl].mean() / c["mu"])
    m["cof_rise_fe"] = rise_frames(c["cof"], pl)
    m["cof_rise_rom"] = rise_frames(cof_h, pl)
    m["Fn_ss_rel"] = sf(abs(Fn_h[pl].mean() - Fn_fe[pl].mean()) / Fn_fe[pl].mean())
    m["p_rel"] = rel_err(P_h, P_fe)
    m["pmax_rel"] = sf(abs(P_h.max() - P_fe.max()) / P_fe.max())
    m["h_max_rel"] = sf(abs(h_ro.max() - h_fe.max()) / h_fe.max())
    m["h_int_rel"] = sf(abs(np.sum(h_ro * dx) - np.sum(h_fe * dx)) / np.sum(h_fe * dx))
    m["h_L2_rel"] = rel_err(h_ro, h_fe)
    m["e_t"] = np.linalg.norm(Xh - Xt, axis=0) / np.linalg.norm(Xt, axis=0).max()   # relative to the peak field norm
    return m, dict(cof=cof_h, Fn=Fn_h, h=h_ro, x=x, h_fe=h_fe, e_t=m["e_t"])


def eval_rom(bundle, c, top):
    t0 = time.perf_counter()
    Q, Xh = predict_state(bundle["model"], bundle["basis"], c)
    cof, Fn, P = predict_outputs(bundle["maps"], bundle["pb"], Q, c, bundle["spec"])
    dt = time.perf_counter() - t0
    m, s = metrics(c, Xh, cof, Fn, P, top)
    r = bundle["model"]["r"]
    Vr = bundle["basis"]["V"][:, :r]
    m["e_proj_floor"] = rel_err(Vr @ (Vr.T @ c["Xt"]), c["Xt"])
    m["rho_A"] = rho_lin(bundle["model"])
    m["stable"] = bool(np.all(np.isfinite(Q)))
    m["t_online_s"] = dt
    return m, s, Xh


def mean_of(rows, key):
    return float(np.mean([m[key] for m in rows]))


# 6. INSIGHTS
def insights(model, name):
    r, p, spec = model["r"], model["p"], model["spec"]
    off = 1 if spec["const"] else 0
    Op = model["Op"]
    A, B = Op[:, off:off + r], Op[:, off + r:off + r + p]
    ev = np.linalg.eigvals(A)
    rho = np.abs(ev).max()
    print(f"\n--- {name} insights (r={r}, lambda={model['lam']:.0e}) ---")
    print(f"one-step training error : {model['onestep']:.3e}")
    print(f"spectral radius of A    : {rho:.4f} ({'stable' if rho < 1 else 'UNSTABLE linear part'})")
    for l in ev[np.argsort(-np.abs(ev))][:3]:
        tau = -1.0 / np.log(abs(l)) if abs(l) < 1 else np.inf
        per = 2 * np.pi / abs(np.angle(l)) if abs(np.angle(l)) > 1e-9 else np.inf
        print(f"   |lambda|={abs(l):.4f}  time constant={tau:8.2f} frames  period={per:8.2f} frames")
    bn = np.linalg.norm(B, axis=0)
    print("input influence (share)  : " + ", ".join(f"{n}={v / bn.sum():.2f}" for n, v in zip(INPUT_NAMES, bn)))
    blocks = []
    if spec["const"]:
        blocks.append(("const", 0, 1))
    blocks += [("A q", off, off + r), ("B u", off + r, off + r + p)]
    if spec["quad"]:
        blocks.append(("H q(x)q", off + r + p, off + r + p + r * (r + 1) // 2))
    D = model["D"]
    tot = np.mean(np.linalg.norm(Op @ D, axis=0))
    print("term shares in q_(k+1)   : " + ", ".join(
        f"{k}={np.mean(np.linalg.norm(Op[:, a:b] @ D[a:b], axis=0)) / tot:.2f}" for k, a, b in blocks))


# 7. SWAP TEST AND SCREENING
def swap_test(cases, sel, rng):
    ids = TRAIN + TEST
    others = [i for i in ids if i != BASELINE]
    names = list(VARIANTS) + ["Param-regression", "Baseline-copy"]
    res = {n: [] for n in names}
    rmax = max(sel[(n, "cv")]["r"] for n in VARIANTS)
    done = tries = 0
    while done < N_SWAP and tries < 300:
        tries += 1
        tr_ids = [BASELINE] + [int(i) for i in rng.choice(others, 4, replace=False)]
        te_ids = [i for i in ids if i not in tr_ids]
        G = np.array([[1, cases[i]["u_raw"][0, 0], cases[i]["u_raw"][1, 0]] for i in tr_ids])
        if np.linalg.matrix_rank(G) < 3:
            continue
        tr, te = [cases[i] for i in tr_ids], [cases[i] for i in te_ids]
        b = make_basis(tr, {}, rmax)
        for n, spec in VARIANTS.items():
            m = fit_rom(b, tr, {}, sel[(n, "cv")]["r"], sel[(n, "cv")]["lam"], spec)
            res[n].append(np.mean([rel_err(predict_state(m, b, c)[1], c["Xt"]) for c in te]))
        res["Param-regression"].append(
            np.mean([rel_err(baseline_regression(tr, c, only_state=True), c["Xt"]) for c in te]))
        res["Baseline-copy"].append(np.mean([rel_err(cases[BASELINE]["Xt"], c["Xt"]) for c in te]))
        done += 1
    return res


def virtual_case(base, ds, Ts):
    t, amp = base["t"], base["amp"]
    return dict(t=t, amp=amp, U1=base["U1"],
                u_raw=np.vstack([np.full_like(t, ds), np.full_like(t, Ts), V_SLIDE * amp]),
                u_gate=np.vstack([ds * amp, Ts * amp, V_SLIDE * amp]))


def screening(cases, train, best, sel, pb, fold_bases, basis, top, npts=9):
    spec, r, lam = VARIANTS[best], sel[(best, "cv")]["r"], sel[(best, "cv")]["lam"]
    bundles = [train_bundle(train, spec, r, lam, top, pb, basis)]
    for hid, fb in fold_bases.items():                       # leave-one-case-out ensemble
        tr = [c for c in train if c["id"] != hid]
        bundles.append(train_bundle(tr, spec, r, lam, top, make_pbasis(tr, top), fb))
    dsv = [c["u_raw"][0, 0] for c in train]; Tsv = [c["u_raw"][1, 0] for c in train]
    dsg = np.linspace(min(dsv + [cases[i]["u_raw"][0, 0] for i in TEST]), max(dsv + [cases[i]["u_raw"][0, 0] for i in TEST]), npts)
    Tsg = np.linspace(min(Tsv), max(Tsv), npts)
    base = cases[BASELINE]; pl = pmask(base["t"]); dx = np.gradient(base["xy"][top, 0])
    out = {k: np.full((len(bundles), npts, npts), np.nan) for k in ("cof", "Fn", "hint")}
    for a, bd in enumerate(bundles):
        for i, ds in enumerate(dsg):
            for j, Ts in enumerate(Tsg):
                vc = virtual_case(base, ds, Ts)
                Q = rollout(bd["model"], get_u(vc, spec["gate"]))
                cof, Fn, P = predict_outputs(bd["maps"], bd["pb"], Q, vc, spec)
                out["cof"][a, i, j] = cof[pl].mean()
                out["Fn"][a, i, j] = Fn[pl].mean()
                out["hint"][a, i, j] = np.sum(wear(P, vc["U1"]) * dx)
    return dsg, Tsg, out



# 9. PAPER OUTPUTS: publication-style figures, tables and a JSON with every reported number

STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
         "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
         "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0, "savefig.dpi": 300,
         "savefig.bbox": "tight"}
COL = {"FE": "k", "DMDc": "#1f77b4", "DMDc+c": "#d62728", "OpInf": "#2ca02c", "DMDc-gated": "#9467bd",
       "OpInf-gated": "#8c564b", "Param-regression": "#7f7f7f", "Baseline-copy": "#bcbd22"}


def _save(fig, name):
    fig.savefig(f"final_{name}.png"); plt.close(fig)


def make_outputs(ctx):
    import json
    plt.rcParams.update(STYLE)
    cases, train, test, top = ctx["cases"], ctx["train"], ctx["test"], ctx["top"]
    basis, en, sel, rows, series = ctx["basis"], ctx["en"], ctx["sel"], ctx["rows"], ctx["series"]
    preds, best, swap, extrap = ctx["preds"], ctx["best"], ctx["swap"], ctx["extrap"]
    scr, dsg, Tsg = ctx["scr"], ctx["dsg"], ctx["Tsg"]
    bcase = cases[BASELINE]
    xs = bcase["xy"][top, 0]
    NF = bcase["t"].shape[0]
    mid = NF // 2

    def agg(variant, mode, key):
        R = [m for m in rows if m["variant"] == variant and m["mode"] == mode]
        return float(np.mean([m[key] for m in R]))

    keys = [k for k in rows[0] if k not in ("variant", "mode", "case", "r", "lam", "cv", "e_t")]
    # ---- CSV ---------------------------------------------------------------------------
    with open("rom_test_metrics.csv", "w", newline="") as f:
        w = csv.writer(f); head = ["variant", "mode", "case", "r", "lam", "cv"] + keys; w.writerow(head)
        for m in rows:
            w.writerow([m.get(k, "") for k in head])

    # ---- Fig. 1: FE verification ------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.0))
    for cid, ls in ((6, ":"), (5, "-"), (4, "--")):
        c = cases[cid]
        ax[0].plot(c["t"], np.abs(c["RF2"]), "k", ls=ls, label=f"{1e3 * c['delta']:.1f} um")
        ax[2].plot(xs, c["P"][top, mid], "k", ls=ls)
    for cid, col in ((2, "#1f77b4"), (5, "k"), (8, "#d62728")):
        c = cases[cid]
        ax[1].plot(c["t"], c["cof"], color=col, label=f"{c['T']} C")
        ax[1].axhline(c["mu"], color=col, lw=0.5, ls=":")
    for T, mk in ((-120, "o"), (25, "s"), (200, "^")):
        ids = [i for i in CASES if CASES[i][1] == T]
        ids.sort(key=lambda i: CASES[i][0])
        ax[3].plot([1e3 * CASES[i][0] for i in ids], [cases[i]["Fn_pl"] for i in ids], "-" + mk, ms=3, label=f"{T} C")
    ax[0].set_xlabel("t (s)"); ax[0].set_ylabel("F_n = |RF2| (N/mm)"); ax[0].legend(title="delta", title_fontsize=6)
    ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("COF = |RF1|/|RF2|"); ax[1].legend(title="T", title_fontsize=6)
    ax[2].set_xlabel("x (mm)"); ax[2].set_ylabel("CPRESS at mid-step (MPa)"); ax[2].set_xlim(0.6, 2.6)
    ax[3].set_xlabel("delta (um)"); ax[3].set_ylabel("plateau F_n (N/mm)"); ax[3].legend()
    for a, l in zip(ax, "abcd"):
        a.set_title(f"({l})", loc="left", fontsize=8)
    fig.tight_layout(); _save(fig, "fig1_fe_verification")

    # ---- Fig. 2: reducibility ----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.0))
    ax[0].semilogy(basis["s"] / basis["s"][0], "k.-", ms=2)
    ax[0].set_xlabel("mode index"); ax[0].set_ylabel("normalised singular value")
    ax[1].semilogy(np.arange(1, 41), 1 - en[:40], "k.-", ms=3)
    for e, ls in ((0.99, ":"), (0.999, "--"), (0.9999, "-.")):
        rr = int(np.searchsorted(en, e) + 1); ax[1].axvline(rr, color="r", lw=0.6, ls=ls, label=f"{100 * e:g}%: r={rr}")
    ax[1].set_xlabel("number of modes r"); ax[1].set_ylabel("1 - cumulative energy"); ax[1].legend()
    for a, l in zip(ax, "ab"):
        a.set_title(f"({l})", loc="left", fontsize=8)
    fig.tight_layout(); _save(fig, "fig2_reducibility")

    # ---- Fig. 3: model comparison ---------------------------------------------------------
    names = list(VARIANTS) + ["Param-regression", "Baseline-copy"]
    mode = lambda n: "CV-selected r" if n in VARIANTS else ("baseline" if n == "Param-regression" else "null model")
    eu = [agg(n, mode(n), "e_u") for n in names]
    lo = [np.min(swap[n]) for n in names]; hi = [np.max(swap[n]) for n in names]; md = [np.median(swap[n]) for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.15, 1]})
    xx = np.arange(len(names))
    ax[0].bar(xx, eu, color=[COL[n] for n in names], edgecolor="k", lw=0.4)
    ax[0].errorbar(xx, md, yerr=[np.array(md) - np.array(lo), np.array(hi) - np.array(md)], fmt="k_", capsize=2, lw=0.8,
                   label="swap test: median, min-max")
    ax[0].set_xticks(xx); ax[0].set_xticklabels([n.replace("Param-regression", "Param.\nregr.").replace("Baseline-copy", "Baseline\ncopy")
                                                  .replace("-gated", "\ngated") for n in names], fontsize=6)
    ax[0].set_ylabel("mean test state error e_u"); ax[0].set_yscale("log"); ax[0].legend(loc="upper left")
    wdt = 0.8 / len(SAME_R)
    for k, rr in enumerate(SAME_R):
        ax[1].bar(np.arange(len(VARIANTS)) + k * wdt, [agg(n, f"fixed r={rr}", "e_u") for n in VARIANTS], wdt,
                  label=f"r = {rr}", edgecolor="k", lw=0.4, color=plt.cm.Greys(0.35 + 0.2 * k))
    ax[1].set_xticks(np.arange(len(VARIANTS)) + wdt); ax[1].set_xticklabels([n.replace("-gated", "\ngated") for n in VARIANTS], fontsize=6)
    ax[1].set_ylabel("mean test e_u at fixed r"); ax[1].legend()
    for a, l in zip(ax, "ab"):
        a.set_title(f"({l})", loc="left", fontsize=8)
    fig.tight_layout(); _save(fig, "fig3_model_comparison")

    # ---- Fig. 4: test outputs ----------------------------------------------------------------
    show = [("DMDc", ("DMDc", "cv")), (best + " (best)", (best, "cv")), ("OpInf", ("OpInf", "cv"))]
    fig, ax = plt.subplots(2, len(test), figsize=(7.2, 3.6))
    for i, c in enumerate(test):
        ax[0, i].plot(c["t"], c["cof"], "k", lw=1.4, label="FE")
        ax[1, i].plot(xs, series[("base", "base", c["id"])]["h_fe"], "k", lw=1.4, label="FE")
        for lab, key in show:
            s_ = series[key + (c["id"],)]
            nm = key[0]
            ax[0, i].plot(c["t"], s_["cof"], "--", color=COL[nm], label=lab)
            ax[1, i].plot(s_["x"], s_["h"], "--", color=COL[nm], label=lab)
        ax[0, i].set_title(f"case {c['id']} (delta={1e3 * c['delta']:.1f} um, T={c['T']} C)", fontsize=7)
        ax[1, i].set_xlim(0.6, 2.6); ax[1, i].set_xlabel("x (mm)"); ax[0, i].set_xlabel("t (s)")
    ax[0, 0].set_ylabel("COF"); ax[1, 0].set_ylabel("wear depth h (mm)"); ax[0, 0].legend(loc="lower right")
    fig.tight_layout(); _save(fig, "fig4_test_outputs")

    # ---- Fig. 5: surface profiles ----------------------------------------------------------------
    c = test[0]; N = c["Xt"].shape[0] // 2
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.1), sharey=True)
    for a, (lab, key) in zip(ax, show):
        Xh = preds[key + (c["id"],)]
        for j, col in zip((25, 50, 75, 100), ("C0", "C1", "C2", "C3")):
            a.plot(xs, 1e3 * c["Xt"][N + top, j], col, lw=0.9)
            a.plot(xs, 1e3 * Xh[N + top, j], col, ls="--", lw=0.9)
        a.set_title(f"{lab}", fontsize=7); a.set_xlabel("x (mm)"); a.set_xlim(-1, 4)
    ax[0].set_ylabel("sliding increment U2 (um)")
    fig.tight_layout(); _save(fig, "fig5_surface_profiles")

    # ---- Fig. 6: screening map (wear rate from F_n) ------------------------------------------------
    dl = (0.5 + 0.3 * dsg) if INPUT_MODE == "delta" else dsg
    Tl = 25 + 175 * Tsg
    med = {k: np.nanmedian(v, axis=0) for k, v in scr.items()}
    spread = 100 * (np.nanmax(scr["Fn"], 0) - np.nanmin(scr["Fn"], 0)) / np.abs(med["Fn"])
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    for a, (Z, ttl) in zip(ax, ((med["cof"], "steady COF"), (K_WEAR * V_SLIDE * med["Fn"], "wear rate k V F_n (mm^2/s)"),
                                (spread, "ensemble spread of F_n (%)"))):
        cs = a.contourf(Tl, dl, Z, 12, cmap="viridis"); fig.colorbar(cs, ax=a, pad=0.02); a.set_title(ttl, fontsize=7)
        a.set_xlabel("T (C)"); a.set_ylabel("delta (um)")
        for cc in train:
            a.plot(25 + 175 * cc["u_raw"][1, 0], 0.5 + 0.3 * cc["u_raw"][0, 0], "wo", mec="k", ms=3)
        for cc in test:
            a.plot(25 + 175 * cc["u_raw"][1, 0], 0.5 + 0.3 * cc["u_raw"][0, 0], "rx", ms=3)
    fig.tight_layout(); _save(fig, "fig6_screening")

    # ---- Fig. S1: error against r with lambda re-tuned at every r --------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    sweep_out = {}
    for nm, spec in VARIANTS.items():
        cvmap = sel[(nm, "cv")]["map"]; pts = []
        for i, rr in enumerate(R_LIST):
            if not np.isfinite(cvmap[i]).any():
                pts.append((rr, np.nan)); continue
            lam = LAMS[int(np.nanargmin(np.where(np.isfinite(cvmap[i]), cvmap[i], np.nan)))]
            mm = fit_rom(basis, train, {}, rr, lam, spec)
            pts.append((rr, np.mean([rel_err(predict_state(mm, basis, c)[1], c["Xt"]) for c in test])))
        pts = np.array(pts); sweep_out[nm] = pts.tolist()
        ax.semilogy(pts[:, 0], np.clip(pts[:, 1], 1e-3, 1e3), "o-", ms=2.5, color=COL[nm], label=nm)
    ax.axhline(agg("Param-regression", "baseline", "e_u"), color="k", ls="--", lw=0.7, label="Param. regression")
    ax.set_xlabel("number of modes r"); ax.set_ylabel("mean test e_u (1e3 = unstable)"); ax.legend(fontsize=5.5)
    fig.tight_layout(); _save(fig, "figS1_error_vs_r")

    # ---- JSON with every number quoted in the paper ------------------------------------------------------
    f_ = lambda v: float(v) if np.isfinite(v) else None
    tab = {}
    for n in names:
        md_ = mode(n)
        tab[n] = {k: f_(agg(n, md_, k)) for k in keys if k not in ("stable",)}
        tab[n]["swap_median"], tab[n]["swap_min"], tab[n]["swap_max"] = f_(np.median(swap[n])), f_(np.min(swap[n])), f_(np.max(swap[n]))
        if n in VARIANTS:
            tab[n].update(r=int(sel[(n, "cv")]["r"]), lam=float(sel[(n, "cv")]["lam"]), cv=f_(sel[(n, "cv")]["cv"]),
                          extrap_trained=f_(extrap[n][0]), extrap_unseen=f_(extrap[n][1]),
                          stable_cases=int(sum(m["stable"] for m in rows if m["variant"] == n and m["mode"] == "CV-selected r")))
    out = dict(best_by_cv=best, table=tab,
               same_r={n: {str(rr): f_(agg(n, f"fixed r={rr}", "e_u")) for rr in SAME_R} for n in VARIANTS},
               audit={str(i): dict(delta_um=1e3 * cases[i]["delta"], T=cases[i]["T"], mu=cases[i]["mu"], Fn=cases[i]["Fn_pl"],
                                   pmax=float(cases[i]["P"].max()), cof_over_mu=float(cases[i]["cof"][pmask(cases[i]["t"])].mean() / cases[i]["mu"]))
                      for i in sorted(cases)},
               pod_energy={str(e): int(np.searchsorted(en, e) + 1) for e in (0.99, 0.999, 0.9999)},
               error_vs_r=sweep_out,
               per_case_best=[{k: f_(m[k]) for k in keys if k != "stable"} | {"case": m["case"]}
                              for m in rows if m["variant"] == best and m["mode"] == "CV-selected r"])
    if ABAQUS_WALL_S:
        out["speedup"] = {n: float(np.mean([ABAQUS_WALL_S[m["case"]] / m["t_online_s"] for m in rows
                                            if m["variant"] == n and m["mode"] == "CV-selected r"])) for n in VARIANTS}
    json.dump(out, open("paper_numbers.json", "w"), indent=1)
    print("\nNULL MODEL (copy of baseline): mean test e_u = %.4f ; best ROM = %.4f ; parametric regression = %.4f" % (
        tab["Baseline-copy"]["e_u"], tab[best]["e_u"], tab["Param-regression"]["e_u"]))
    if ABAQUS_WALL_S:
        print("speed-up vs Abaqus:", {n: round(v) for n, v in out["speedup"].items()})
    else:
        print("Fill ABAQUS_WALL_S = {1: seconds, ..., 9: seconds} to report the speed-up.")


# 8. MAIN
def main():
    rng = np.random.default_rng(SEED)
    cases = {cid: load_case(cid) for cid in TRAIN + TEST}
    build_inputs(cases)
    check_consistency(cases)
    train, test = [cases[i] for i in TRAIN], [cases[i] for i in TEST]
    top = top_nodes(cases[TRAIN[0]]["xy"])
    n, nf = cases[TRAIN[0]]["Xt"].shape
    print(f"state dimension n = {n}, frames = {nf}, surface nodes = {len(top)}, INPUT_MODE = {INPUT_MODE}")
    data_audit(cases, top)

    dtr, dte = combine(train), combine(test)
    np.savez_compressed("rom_dataset.npz", X_train=dtr["Xt"], U_train=dtr["U"], case_train=dtr["case_id"],
                        frame_train=dtr["frame"], X0_train=dtr["x0"], X_test=dte["Xt"], U_test=dte["U"],
                        case_test=dte["case_id"], frame_test=dte["frame"], X0_test=dte["x0"], top_nodes=top)
    print(f"X_train {dtr['Xt'].shape}, X_test {dte['Xt'].shape} -> rom_dataset.npz")

    rmax = max(R_LIST + SAME_R)
    basis = make_basis(train, {}, rmax)
    en = np.cumsum(basis["s"] ** 2) / np.sum(basis["s"] ** 2)
    print("POD energy: " + ", ".join(f"r({100 * e:g}%)={int(np.searchsorted(en, e) + 1)}" for e in (0.99, 0.999, 0.9999)))
    pb = make_pbasis(train, top)
    fold_bases = make_fold_bases(train, rmax)

    # ---- model selection (CV only; test cases are never used here) ------------------
    sel = {}
    for name, spec in VARIANTS.items():
        t0 = time.perf_counter()
        r, lam, cv, cvmap = cv_select(train, cases, spec, R_LIST, LAMS, fold_bases, basis)
        sel[(name, "cv")] = dict(r=r, lam=lam, cv=cv, map=cvmap)
        for rr in SAME_R:
            r2, lam2, cv2, _ = cv_select(train, cases, spec, [rr], LAMS, fold_bases, basis)
            sel[(name, rr)] = dict(r=r2, lam=lam2, cv=cv2)
        print(f"selected {name:12s}: r={r:2d} lambda={lam:.0e} CV error={cv:.3f}  ({time.perf_counter() - t0:.0f}s)")
        if lam == LAMS[-1] or lam == LAMS[0]:
            print("   NOTE: lambda is at the edge of the grid; widen LAMS")
    best = min(VARIANTS, key=lambda nm: sel[(nm, "cv")]["cv"])
    print(f"best variant by CV (not by test): {best}")

    # ---- training and test evaluation -------------------------------------------------
    rows, series, preds, bundles = [], {}, {}, {}
    for (name, key), s in sel.items():
        bd = train_bundle(train, VARIANTS[name], s["r"], s["lam"], top, pb, basis)
        bundles[(name, key)] = bd
        for c in test:
            m, ser, Xh = eval_rom(bd, c, top)
            m.update(variant=name, mode=("CV-selected r" if key == "cv" else f"fixed r={key}"),
                     r=s["r"], lam=s["lam"], cv=s["cv"])
            rows.append(m); series[(name, key, c["id"])] = ser; preds[(name, key, c["id"])] = Xh
    base_rows = []
    for c in test:
        Xh, cof, Fn, P = baseline_regression(train, c, top=top)
        m, ser = metrics(c, Xh, cof, Fn, P, top)
        m.update(variant="Param-regression", mode="baseline", r=np.nan, lam=np.nan, cv=np.nan,
                 e_proj_floor=np.nan, rho_A=np.nan, stable=True, t_online_s=np.nan)
        base_rows.append(m); rows.append(m); series[("base", "base", c["id"])] = ser
        preds[("base", "base", c["id"])] = Xh

    b0 = cases[BASELINE]                              # null model: ignore delta and T, copy the baseline
    copy_rows = []
    for c in test:
        m, ser = metrics(c, b0["Xt"], b0["cof"], np.abs(b0["RF2"]), b0["P"][top], top)
        m.update(variant="Baseline-copy", mode="null model", r=np.nan, lam=np.nan, cv=np.nan,
                 e_proj_floor=np.nan, rho_A=np.nan, stable=True, t_online_s=np.nan)
        copy_rows.append(m); rows.append(m)

    for nm in ("DMDc", "OpInf") + ((best,) if best not in ("DMDc", "OpInf") else ()):
        insights(bundles[(nm, "cv")]["model"], nm)
        print("training rollout error per case: " + ", ".join(
            f"C{c['id']}={rel_err(predict_state(bundles[(nm, 'cv')]['model'], basis, c)[1], c['Xt']):.3f}" for c in train))

    # ---- tables -------------------------------------------------------------------------
    def pick(variant, mode):
        return [m for m in rows if m["variant"] == variant and m["mode"] == mode]

    cols = ["e_u", "e_surf", "e_far_share", "cof_ss_abs", "cof_rmse_trans", "Fn_ss_rel", "h_int_rel", "h_L2_rel"]
    print("\n" + "=" * 118 + "\nTABLE A: each variant with its CV-selected (r, lambda). Mean over the 4 test cases\n" + "=" * 118)
    print(f"{'variant':16s}{'r':>3} {'lambda':>7} {'CV':>7} " + " ".join(f"{k[:11]:>11}" for k in cols) + f" {'rho(A)':>7} {'stable':>6}")
    for nm in VARIANTS:
        R = pick(nm, "CV-selected r")
        s = sel[(nm, "cv")]
        print(f"{nm:16s}{s['r']:>3} {s['lam']:7.0e} {s['cv']:7.3f} " + " ".join(f"{mean_of(R, k):11.3g}" for k in cols)
              + f" {mean_of(R, 'rho_A'):7.3f} {sum(m['stable'] for m in R):>4}/4")
    print(f"{'Param-regression':16s}{'-':>3} {'-':>7} {'-':>7} " + " ".join(f"{mean_of(base_rows, k):11.3g}" for k in cols))
    print(f"{'Baseline-copy':16s}{'-':>3} {'-':>7} {'-':>7} " + " ".join(f"{mean_of(copy_rows, k):11.3g}" for k in cols)
          + "   <- null model: a ROM must beat this to show it learned any parameter dependence")

    print("\n" + "=" * 118 + "\nTABLE B: same-r comparison (fair). Mean over the 4 test cases: e_u / steady-COF error / integrated-wear error / stable\n" + "=" * 118)
    print(f"{'variant':16s}" + "".join(f"{'r=' + str(rr):>34}" for rr in SAME_R))
    for nm in VARIANTS:
        line = f"{nm:16s}"
        for rr in SAME_R:
            R = pick(nm, f"fixed r={rr}")
            line += f"{mean_of(R, 'e_u'):9.3f} /{mean_of(R, 'cof_ss_abs'):7.4f} /{mean_of(R, 'h_int_rel'):7.3f} /{sum(m['stable'] for m in R)}/4 "
        print(line)

    print("\nABLATION READING (mean e_u at each variant's CV-selected r; smaller is better)")
    eu = {nm: mean_of(pick(nm, "CV-selected r"), "e_u") for nm in VARIANTS}
    for a, b, what in (("DMDc", "DMDc+c", "adding the constant term"),
                       ("DMDc", "DMDc-gated", "gating the inputs by the speed ramp"),
                       ("DMDc+c", "OpInf", "adding the quadratic term (with constant)"),
                       ("DMDc-gated", "OpInf-gated", "adding the quadratic term (gated)")):
        print(f"  {what:46s}: {a} {eu[a]:.3f} -> {b} {eu[b]:.3f}")
    print(f"  best ROM variant {eu[best]:.3f} | parametric regression {mean_of(base_rows, 'e_u'):.3f} | "
          f"baseline-copy (null) {mean_of(copy_rows, 'e_u'):.3f}")

    print("\nPer-case results for the best variant (" + best + ")")
    print(f"{'case':>5} " + " ".join(f"{k[:11]:>11}" for k in cols + ['cof_rise_fe', 'cof_rise_rom']))
    for m in pick(best, "CV-selected r"):
        print(f"{m['case']:>5} " + " ".join(f"{m[k]:11.3g}" for k in cols + ['cof_rise_fe', 'cof_rise_rom']))

    if ABAQUS_WALL_S:
        for nm in VARIANTS:
            sp = np.mean([ABAQUS_WALL_S[m["case"]] / m["t_online_s"] for m in pick(nm, "CV-selected r")])
            print(f"online speed-up vs Abaqus, {nm}: {sp:.0f}x")

    # ---- time extrapolation ----------------------------------------------------------
    print("\nTime-extrapolation (baseline case; trained on frames 0-70 only)")
    fr = {BASELINE: EXTRAP_FRAMES}
    extrap = {}
    be = make_basis(train, fr, rmax)
    Xt = cases[BASELINE]["Xt"]
    for nm, spec in VARIANTS.items():
        s = sel[(nm, "cv")]
        me = fit_rom(be, train, fr, s["r"], s["lam"], spec)
        Xh = predict_state(me, be, cases[BASELINE])[1]
        extrap[nm] = (rel_err(Xh[:, :EXTRAP_FRAMES], Xt[:, :EXTRAP_FRAMES]), rel_err(Xh[:, EXTRAP_FRAMES:], Xt[:, EXTRAP_FRAMES:]))
        print(f"  {nm:12s}: error on trained frames {rel_err(Xh[:, :EXTRAP_FRAMES], Xt[:, :EXTRAP_FRAMES]):.3f}, "
              f"unseen frames {rel_err(Xh[:, EXTRAP_FRAMES:], Xt[:, EXTRAP_FRAMES:]):.3f}")

    # ---- swap test ----------------------------------------------------------------------
    print(f"\nSwap test: {N_SWAP} other 5/4 splits (baseline always in training); mean test e_u per split")
    swap = swap_test(cases, sel, rng)
    for nm, v in swap.items():
        v = np.array(v)
        fin = np.isfinite(v)
        print(f"  {nm:17s}: median {np.median(v):7.3f}   min {v.min():7.3f}   max {v.max():8.3f}   "
              f"stable in {fin.sum()}/{len(v)} splits")

    # ---- screening ---------------------------------------------------------------------
    dsg, Tsg, scr = screening(cases, train, best, sel, pb, fold_bases, basis, top)

    make_outputs(dict(cases=cases, train=train, test=test, top=top, basis=basis, en=en, sel=sel, rows=rows,
                      series=series, preds=preds, base_rows=base_rows, copy_rows=copy_rows, best=best, swap=swap,
                      extrap=extrap, scr=scr, dsg=dsg, Tsg=Tsg, pb=pb))
    print("\nsaved: rom_dataset.npz, rom_test_metrics.csv, paper_numbers.json, final_fig*.png")


if __name__ == "__main__":
    main()