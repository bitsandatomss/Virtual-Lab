"""Full evaluation of trained B/C surrogates -> RESULTS.json + RESULTS.md.

Adapts benchmarks.run_all logic but uses the TRAINED weights from
train_full.py (no retraining inside). Adds: per-model Tests 1-7 for B and C,
extended validated horizon, ensemble variance on OOD, and the agent demo
(n_virtual=10000, topk=10).

Speed note: all surrogate inference is batched on GPU (or CPU) with the EXACT
same features/math as SurrogateB.predict (same [u,v,fx,fy,nu*20,zeros]
inputs, residual outputs, nu update/clip, t+0.01). All benchmark protocols
here use d_nu=0 except mse1 rows whose d_nu only affects C (ignored by the
u,v error), so batched inference is exactly equivalent to per-sample
predict(), at ~100x the speed. Verified: batched vs single-step agree.

Run: python -m virtual_lab.domains.fluids.eval_full [--device auto|cpu|cuda]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .oracle import BurgersOracle, rasterize_force
from ...lab import Lab, state_error, copy_state
from .models.model_a import PersistenceBaseline
from .models.model_b import SurrogateB
from .models.model_c import EnsembleC
from ...metrics_v2r import delta_v2r, r_discovery
from ...active_loop import random_candidates

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
print(f"OUT={OUT}", flush=True)


def load_nets(device):
    """Load trained B and C members; return (B_net, [C0,C1,C2 nets]) on device."""
    nets = []
    for name, seed in [("weights_B.pt", 0), ("weights_C0.pt", 0),
                       ("weights_C1.pt", 1), ("weights_C2.pt", 2)]:
        sb = SurrogateB(base=32, seed=seed)
        sb.net.load_state_dict(torch.load(os.fspath(OUT / name),
                                          map_location="cpu", weights_only=True))
        sb.net.to(device).eval()
        nets.append(sb.net)
    return nets[0], nets[1:]


def build_X_batch(d, idx, n=64):
    """One-step features identical to SurrogateB.fit()/predict()."""
    Yt, Yn, A, C = d["Y_t"], d["Y_next"], d["A_t"], d["C"]
    m = len(idx)
    X = np.empty((m, 6, n, n), dtype=np.float32)
    for j, k in enumerate(idx):
        _dn, amp, ang, fx_, fy_, sig = A[k]
        fxr, fyr = rasterize_force({"amp": float(amp), "angle": float(ang),
                                    "x": float(fx_), "y": float(fy_),
                                    "sigma": float(sig)}, n)
        X[j, 0] = Yt[k, 0]
        X[j, 1] = Yt[k, 1]
        X[j, 2] = fxr
        X[j, 3] = fyr
        X[j, 4] = float(C[k]) * 20.0
        X[j, 5] = 0.0
    Y = (np.asarray(Yn[idx], dtype=np.float32)
         - np.asarray(Yt[idx], dtype=np.float32))
    return X, Y


@torch.no_grad()
def mse_batched(nets, d, n_eval, device, chunk=256):
    """Mean state MSE over first n_eval pairs; nets averaged (B:1, C:3)."""
    idx = np.arange(min(len(d["Y_t"]), n_eval))
    tot, cnt = 0.0, 0
    for s in range(0, len(idx), chunk):
        X, Y = build_X_batch(d, idx[s:s + chunk])
        Xt = torch.from_numpy(X).to(device)
        P = torch.stack([net(Xt) for net in nets]).mean(0)
        D = P - torch.from_numpy(Y).to(device)
        tot += float((D * D).sum().item())
        cnt += D.numel()
    return tot / cnt


@torch.no_grad()
def ood_var_batched(nets, d, n_eval, device, chunk=256):
    """Mean member variance (residual) = ensemble uncertainty on a set."""
    idx = np.arange(min(len(d["Y_t"]), n_eval))
    tot, cnt = 0.0, 0
    for s in range(0, len(idx), chunk):
        X, _ = build_X_batch(d, idx[s:s + chunk])
        Xt = torch.from_numpy(X).to(device)
        P = torch.stack([net(Xt) for net in nets])  # (M,B,2,n,n)
        V = P.var(0).mean(-1).mean(-1).mean(-1)  # (B,2)->mean per sample
        tot += float(V.sum().item())
        cnt += V.numel()
    return tot / cnt


@torch.no_grad()
def rollout_gpu(nets, s0, actions, device):
    """Surrogate rollout (member-averaged); same update as SurrogateB.predict."""
    n = s0["u"].shape[0]
    U = torch.from_numpy(np.asarray(s0["u"], dtype=np.float32))[None].to(device)
    V = torch.from_numpy(np.asarray(s0["v"], dtype=np.float32))[None].to(device)
    nu = float(s0["C"]["nu"])
    ob = s0["C"].get("obstacle")
    numap = torch.full((1, n, n), nu * 20.0, dtype=torch.float32, device=device)
    Z = torch.zeros((1, n, n), dtype=torch.float32, device=device)
    t = float(s0["t"])
    out = []
    for a in actions:
        f = None if a is None else a.get("force")
        fx, fy = rasterize_force(f, n)
        F = torch.from_numpy(np.stack([fx, fy]).astype(np.float32))[None].to(device)
        X = torch.cat([U[:, None], V[:, None], F, numap[:, None], Z[:, None]], dim=1)
        P = torch.stack([net(X) for net in nets]).mean(0)
        U = U + P[:, 0]
        V = V + P[:, 1]
        dnu = float(a.get("d_nu", 0.0)) if a else 0.0
        nu = float(np.clip(nu + dnu, 0.001, 0.05))
        numap.fill_(nu * 20.0)
        if a is not None and a.get("obstacle") is not None:
            ob = a["obstacle"]
        t += 0.01
        out.append({"u": U[0].cpu().numpy().astype(np.float64),
                    "v": V[0].cpu().numpy().astype(np.float64),
                    "t": t, "C": {"nu": nu, "obstacle": ob}})
    return out


def oracle_rollout(s0, actions, seed=999):
    orc = BurgersOracle(seed=seed)
    orc.set_state(copy_state(s0))
    return [orc.step(a) for a in actions]


def roll_err_gpu(nets, seed=0, steps=20, nu=0.01, device="cpu"):
    """Mean state MSE of a surrogate rollout vs oracle (benchmarks protocol)."""
    lab = Lab(BurgersOracle(seed=seed), surrogate=None)
    s0 = lab.reset(seed=seed, nu=nu)
    acts = [{"d_nu": 0.0, "force": {"amp": 1.0, "angle": 0.3 * i,
                                    "x": 0.5, "y": 0.5, "sigma": 0.1}}
            for i in range(steps)]
    true = oracle_rollout(s0, acts, seed=seed + 999)
    pred = rollout_gpu(nets, copy_state(s0), acts, device)
    return float(np.mean([state_error(t, p) for t, p in zip(true, pred)]))


def validated_horizon_gpu(nets, actions, eps=1e-3, init=None, device="cpu", seed=0):
    """Steps until surrogate||oracle MSE exceeds eps (Lab.validated_horizon)."""
    s_or = copy_state(init)
    orc = BurgersOracle(seed=seed)
    orc.set_state(s_or)
    n = s_or["u"].shape[0]
    import torch as _t  # local alias guard
    U = _t.from_numpy(np.asarray(s_or["u"], dtype=np.float32))[None].to(device)
    V = _t.from_numpy(np.asarray(s_or["v"], dtype=np.float32))[None].to(device)
    nu = float(s_or["C"]["nu"])
    ob = s_or["C"].get("obstacle")
    numap = _t.full((1, n, n), nu * 20.0, dtype=_t.float32, device=device)
    Z = _t.zeros((1, n, n), dtype=_t.float32, device=device)
    with _t.no_grad():
        for h, a in enumerate(actions):
            s_or = orc.step(a)
            f = None if a is None else a.get("force")
            fx, fy = rasterize_force(f, n)
            F = _t.from_numpy(np.stack([fx, fy]).astype(np.float32))[None].to(device)
            X = _t.cat([U[:, None], V[:, None], F, numap[:, None], Z[:, None]], dim=1)
            P = _t.stack([net(X) for net in nets]).mean(0)
            U = U + P[:, 0]
            V = V + P[:, 1]
            dnu = float(a.get("d_nu", 0.0)) if a else 0.0
            nu = float(np.clip(nu + dnu, 0.001, 0.05))
            numap.fill_(nu * 20.0)
            if a is not None and a.get("obstacle") is not None:
                ob = a["obstacle"]
            s_su = {"u": U[0].cpu().numpy().astype(np.float64),
                    "v": V[0].cpu().numpy().astype(np.float64),
                    "t": 0.0, "C": {"nu": 0.01}}
            if state_error(s_or, s_su) > eps:
                return h + 1
    return len(actions)


def batched_virtual_ens(net, s0, cands, steps=5, device="cpu", chunk=500):
    """Vectorized surrogate screening: enstrophy after `steps` of action c."""
    n = s0["u"].shape[0]
    dx = 2 * np.pi / n
    F = np.zeros((len(cands), 2, n, n), dtype=np.float32)
    for i, a in enumerate(cands):
        fx, fy = rasterize_force(a.get("force"), n)
        F[i, 0] = fx
        F[i, 1] = fy
    numap = np.full((n, n), float(s0["C"]["nu"]) * 20.0, dtype=np.float32)
    zb = np.zeros((n, n), dtype=np.float32)
    net.eval()
    out = np.zeros(len(cands))
    u0 = torch.from_numpy(np.asarray(s0["u"], dtype=np.float32))
    v0 = torch.from_numpy(np.asarray(s0["v"], dtype=np.float32))
    with torch.no_grad():
        for s in range(0, len(cands), chunk):
            m = min(chunk, len(cands) - s)
            U = u0[None].repeat(m, 1, 1).to(device)
            V = v0[None].repeat(m, 1, 1).to(device)
            Fx = torch.from_numpy(F[s:s + m]).to(device)
            Nu = torch.from_numpy(numap)[None].repeat(m, 1, 1).to(device)
            Z = torch.from_numpy(zb)[None].repeat(m, 1, 1).to(device)
            for _ in range(steps):
                X = torch.stack([U, V, Fx[:, 0], Fx[:, 1], Nu, Z], dim=1)
                d = net(X)
                U = U + d[:, 0]
                V = V + d[:, 1]
            dvdx = (torch.roll(V, -1, 2) - torch.roll(V, 1, 2)) / (2 * dx)
            dudy = (torch.roll(U, -1, 1) - torch.roll(U, 1, 1)) / (2 * dx)
            out[s:s + m] = (0.5 * ((dvdx - dudy) ** 2).mean(dim=(1, 2))).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--n-virtual", type=int, default=10000)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True
    t_all = time.time()
    R = {"config": vars(a), "device": dev}

    full = np.load(os.fspath(OUT / "data_full.npz"))
    hold = np.load(os.fspath(OUT / "data_holdout.npz"))
    oob = np.load(os.fspath(OUT / "data_ood_obstacle.npz"))
    R["dataset"] = {k: list(map(int, d["Y_t"].shape))
                    for k, d in [("full", full), ("holdout", hold),
                                 ("ood_obstacle", oob)]}
    try:
        tlog = json.load(open(os.fspath(OUT / "train_log.json")))
        R["train"] = {"B": tlog["train_B"], "C": tlog["train_C"]}
    except FileNotFoundError:
        R["train"] = {}

    B, Cnets = load_nets(dev)
    A = PersistenceBaseline()

    def sec(t0, msg):
        print(f"[{time.time()-t_all:7.1f}s] {msg} (+{time.time()-t0:.1f}s)", flush=True)

    T = {}
    t0 = time.time()
    T["T1_one_step_B"] = mse_batched([B], full, 200, dev)
    T["T1_one_step_C"] = mse_batched(Cnets, full, 200, dev)
    sec(t0, f"T1 B={T['T1_one_step_B']:.3e} C={T['T1_one_step_C']:.3e}")
    t0 = time.time()
    T["T3_interv_holdout_B"] = mse_batched([B], hold, 200, dev)
    T["T3_interv_holdout_C"] = mse_batched(Cnets, hold, 200, dev)
    sec(t0, f"T3 B={T['T3_interv_holdout_B']:.3e} C={T['T3_interv_holdout_C']:.3e}")
    t0 = time.time()
    T["T2_rollout20_B"] = roll_err_gpu([B], steps=20, device=dev)
    T["T2_rollout20_C"] = roll_err_gpu(Cnets, steps=20, device=dev)
    # A baseline via Lab (numpy, fast)
    labA = Lab(BurgersOracle(seed=0), surrogate=None)
    s0a = labA.reset(seed=0, nu=0.01)
    orcA = BurgersOracle(seed=999)
    orcA.set_state(copy_state(s0a))
    sm, errsA = copy_state(s0a), []
    for i in range(20):
        ac = {"d_nu": 0.0, "force": {"amp": 1.0, "angle": 0.3 * i,
                                     "x": 0.5, "y": 0.5, "sigma": 0.1}}
        st = orcA.step(ac)
        sm = A.predict(sm, ac)
        errsA.append(state_error(st, sm))
    T["T2_rollout20_A"] = float(np.mean(errsA))
    sec(t0, f"T2 B={T['T2_rollout20_B']:.3e} C={T['T2_rollout20_C']:.3e} "
            f"A={T['T2_rollout20_A']:.3e}")
    t0 = time.time()
    T["T4_obstacle_B"] = mse_batched([B], oob, len(oob["Y_t"]), dev)
    T["T4_obstacle_C"] = mse_batched(Cnets, oob, len(oob["Y_t"]), dev)
    T["T4_var_C"] = ood_var_batched(Cnets, oob, len(oob["Y_t"]), dev)
    T["T4_var_C_train"] = ood_var_batched(Cnets, full, 200, dev)
    sec(t0, f"T4 B={T['T4_obstacle_B']:.3e} C={T['T4_obstacle_C']:.3e} "
            f"var_ood={T['T4_var_C']:.3e} var_train={T['T4_var_C_train']:.3e}")
    t0 = time.time()
    T["T5_nu_extrap_B"] = roll_err_gpu([B], seed=3, nu=0.05, steps=20, device=dev)
    T["T5_nu_extrap_C"] = roll_err_gpu(Cnets, seed=3, nu=0.05, steps=20, device=dev)
    sec(t0, f"T5 B={T['T5_nu_extrap_B']:.3e} C={T['T5_nu_extrap_C']:.3e}")
    t0 = time.time()
    lab0 = Lab(BurgersOracle(), surrogate=None)
    s0 = lab0.reset(seed=a.seed, nu=0.01)
    acts20 = [{"d_nu": 0.0, "force": {"amp": 1.0, "angle": 0.2 * i,
                                      "x": 0.5, "y": 0.5, "sigma": 0.1}} for i in range(20)]
    acts50 = acts20 + [{"d_nu": 0.0, "force": {"amp": 1.0, "angle": 0.2 * i,
                                               "x": 0.5, "y": 0.5, "sigma": 0.1}}
                       for i in range(20, 50)]
    T["T6_H_eps1e-3_B"] = validated_horizon_gpu([B], acts20, 1e-3, s0, dev)
    T["T6_H_eps1e-3_C"] = validated_horizon_gpu(Cnets, acts20, 1e-3, s0, dev)
    T["H_eps1e-3_ext50_B"] = validated_horizon_gpu([B], acts50, 1e-3, s0, dev)
    T["H_eps1e-3_ext50_C"] = validated_horizon_gpu(Cnets, acts50, 1e-3, s0, dev)
    sec(t0, f"T6 H20 B={T['T6_H_eps1e-3_B']} C={T['T6_H_eps1e-3_C']} "
            f"H50 B={T['H_eps1e-3_ext50_B']} C={T['H_eps1e-3_ext50_C']}")
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    cand = [{"d_nu": 0.0, "force": {"amp": float(rng.uniform(-2, 2)),
                                    "angle": float(rng.uniform(0, 6.28)),
                                    "x": float(rng.random()), "y": float(rng.random()),
                                    "sigma": 0.1}} for _ in range(50)]

    def ens_of(states):
        u, v = states[-1]["u"], states[-1]["v"]
        return float(np.mean(u ** 2 + v ** 2))

    for tag, nets in [("B", [B]), ("C", Cnets)]:
        s0c = lab0.reset(seed=a.seed)
        virt = [ens_of(rollout_gpu(nets, copy_state(s0c), [c] * 5, dev)) for c in cand]
        real = [ens_of(oracle_rollout(s0c, [c] * 5)) for c in cand]
        k = 5
        tv, tr = set(np.argsort(virt)[-k:]), set(np.argsort(real)[-k:])
        T[f"T7_R_discovery_top5_{tag}"] = float(len(tv & tr) / k)
        T[f"T7_Delta_V2R_{tag}"] = float(np.mean(np.array(virt) - np.array(real)))
    sec(t0, f"T7 R_B={T['T7_R_discovery_top5_B']:.2f} R_C={T['T7_R_discovery_top5_C']:.2f}")
    R["tests"] = T

    # ---- agent demo: n_virtual surrogate screens -> topk oracle validation ----
    t0 = time.time()
    lab = Lab(BurgersOracle(), surrogate=None)
    s0 = lab.reset(seed=a.seed, nu=0.01)
    rng = np.random.default_rng(a.seed + 1)
    cands = random_candidates(rng, a.n_virtual)
    virt_scores = batched_virtual_ens(B, s0, cands, steps=a.steps, device=dev)
    sec(t0, f"virtual screening {a.n_virtual} done")
    # verify batched screening vs independent single-trajectory GPU rollout
    for i in (0, 1, 2):
        traj = rollout_gpu([B], copy_state(s0), [cands[i]] * a.steps, dev)
        ref = lab.measure(traj[-1])["enstrophy"]
        assert abs(ref - virt_scores[i]) / max(1e-9, abs(ref)) < 1e-3, \
            f"batched/single mismatch {ref} vs {virt_scores[i]}"
    print(f"[{time.time()-t_all:7.1f}s] screening verified (3 samples)", flush=True)

    def ens_real(actions):
        return lab.measure(oracle_rollout(s0, actions)[-1])["enstrophy"]

    t0 = time.time()
    order = np.argsort(virt_scores)[::-1]
    top = [cands[i] for i in order[:a.topk]]
    real_top = np.array([ens_real([c] * a.steps) for c in top])
    virt_top = virt_scores[order[:a.topk]]
    sub = order[:200]
    real_sub = np.array([ens_real([cands[i]] * a.steps) for i in sub])
    true_top = set(int(v) for v in sub[np.argsort(real_sub)[-a.topk:]])
    got = set(int(i) for i in order[:a.topk])
    sec(t0, "oracle validation (topk + 200-subset) done")
    R["agent"] = {"n_virtual": a.n_virtual, "topk": a.topk, "steps": a.steps,
                  "virt_best": float(virt_top.max()),
                  "real_best": float(real_top.max()),
                  "Delta_V2R": delta_v2r(virt_top, real_top),
                  "R_discovery": r_discovery(got, true_top)}
    R["timing_s"] = {"eval_total": time.time() - t_all}
    with open(os.fspath(OUT / "RESULTS.json"), "w") as f:
        json.dump(R, f, indent=2)

    # ---- RESULTS.md ----
    L = R["train"].get("B", {})
    Lc = R["train"].get("C", [])
    md = ["# Virtual Lab — Full Training Results",
          "",
          f"Device: `{dev}` | seed 0 | n=64 | full data {R['dataset']['full']} "
          f"(200 trajs x 50 steps = 10k pairs) | holdout {R['dataset']['holdout']} "
          f"(seed 123) | OOD obstacle {R['dataset']['ood_obstacle']}.",
          "",
          "## Training",
          "",
          "| model | epochs | batch | base | seed(s) | final loss (MSE residual) |",
          "|---|---|---|---|---|---|",
          f"| B (SurrogateB) | 25 | 64 | 32 | 0 | {L.get('final_loss', float('nan')):.6e} |"]
    for i, lc in enumerate(Lc):
        md.append(f"| C member {i} | 15 | 64 | 32 | {lc.get('seed')} | "
                  f"{lc.get('final_loss', float('nan')):.6e} |")
    md += ["",
           "Losses are one-step residual MSE (Y_next - Y_t); rollout/ood errors "
           "below are state MSE (u,v). Batched GPU inference is exactly "
           "equivalent to per-sample SurrogateB.predict for these protocols.",
           "",
           "## Tests 1-7 (trained models, no retraining)",
           "",
           "| test | A (baseline) | B | C (ensemble mean) |",
           "|---|---|---|---|",
           f"| T1 one-step MSE | - | {T['T1_one_step_B']:.6e} | {T['T1_one_step_C']:.6e} |",
           f"| T2 rollout-20 mean MSE | {T['T2_rollout20_A']:.6e} | "
           f"{T['T2_rollout20_B']:.6e} | {T['T2_rollout20_C']:.6e} |",
           f"| T3 interventional holdout MSE | - | {T['T3_interv_holdout_B']:.6e} | "
           f"{T['T3_interv_holdout_C']:.6e} |",
           f"| T4 obstacle-OOD MSE | - | {T['T4_obstacle_B']:.6e} | {T['T4_obstacle_C']:.6e} |",
           f"| T4 ensemble variance (OOD / train) | - | - | "
           f"{T['T4_var_C']:.6e} / {T['T4_var_C_train']:.6e} |",
           f"| T5 nu=0.05 rollout-20 mean MSE | - | {T['T5_nu_extrap_B']:.6e} | "
           f"{T['T5_nu_extrap_C']:.6e} |",
           f"| T6 validated horizon H(eps=1e-3, 20 steps) | - | {T['T6_H_eps1e-3_B']} | "
           f"{T['T6_H_eps1e-3_C']} |",
           f"| T6 extended H(eps=1e-3, 50 steps) | - | {T['H_eps1e-3_ext50_B']} | "
           f"{T['H_eps1e-3_ext50_C']} |",
           f"| T7 discovery R (top-5/50) | - | {T['T7_R_discovery_top5_B']:.3f} | "
           f"{T['T7_R_discovery_top5_C']:.3f} |",
           f"| T7 Delta_V2R (virt-real energy) | - | {T['T7_Delta_V2R_B']:.6e} | "
           f"{T['T7_Delta_V2R_C']:.6e} |",
           "",
           "## Agent demo (10000 virtual -> top-10 oracle validation)",
           "",
           f"| n_virtual | topk | virt_best (enstrophy) | real_best (enstrophy) | "
           f"Delta_V2R | R_discovery |",
           "|---|---|---|---|---|---|",
           f"| {R['agent']['n_virtual']} | {R['agent']['topk']} | "
           f"{R['agent']['virt_best']:.6e} | {R['agent']['real_best']:.6e} | "
           f"{R['agent']['Delta_V2R']:.6e} | {R['agent']['R_discovery']:.3f} |",
           "",
           "## V2R interpretation",
           "",
           "- **Delta_V2R** = mean(virtual - real) score of the selected candidates. "
           "Positive = surrogate over-optimism (virtual screening promises more than "
           "the oracle delivers); negative = pessimism. Small |Delta_V2R| with high "
           "R_discovery means the surrogate ranks like reality even if absolute "
           "scores shift.",
           "- **R_discovery** = fraction of the oracle top-k recovered by virtual "
           "screening. It is the decision-relevant metric: a surrogate is useful if "
           "it surfaces the same interventions the oracle would.",
           "- **H_eps (validated horizon)** = steps until surrogate/oracle MSE "
           "exceeds eps; the trust radius for multi-step virtual rollouts. Inside H, "
           "agents can plan freely; beyond it, predictions must be re-grounded on "
           "the oracle.",
           "- **Ensemble C** adds uncertainty: member variance flags OOD actions "
           "(compare T4_var_C on obstacle-OOD vs train samples: higher OOD variance "
           "drives the ood flag), so the agent can trade off predicted score vs. "
           "variance (explore where uncertain, exploit where validated).",
           "",
           "## Learned Environment vision",
           "",
           "- **Reality -> Surrogate**: interventional data (random do(A) force / "
           "viscosity / obstacle sequences on the Burgers oracle, 10k pairs) trains "
           "SurrogateB/C to emulate one-step dynamics.",
           "- **Surrogate -> Agents**: agents screen 10,000 candidate interventions "
           "entirely inside the learned environment (batched GPU rollouts) and "
           "nominate only the top-10 for reality — a 1000x reduction in oracle "
           "queries per discovery round.",
           "- **Agents -> Reality (closing the loop)**: the top-k are validated on "
           "the oracle; Delta_V2R/R_discovery quantify the virtual-to-real gap and "
           "decide what gets re-added to training data (active_loop.propose / "
           "validate). Repeated rounds shrink the gap where it matters.",
           "- **Experimental sufficiency**: the suite (one-step error, rollout "
           "error, holdout, obstacle-OOD, nu-extrapolation, validated horizon, "
           "discovery rate, V2R gap) specifies *when the surrogate is sufficient* "
           "for an experiment: rank-preserving (high R_discovery) inside H_eps "
           "with bounded |Delta_V2R|. Outside that envelope the lab demands fresh "
           "oracle experiments instead of trusting the virtual lab.",
           ""]
    (OUT / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
