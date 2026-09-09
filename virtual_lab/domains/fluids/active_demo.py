"""Active loop demo: uncertainty-weighted vs random acquisition.

The core loop: surrogate proposes candidates virtually, oracle validates
the top-k. Two acquisition strategies compared:
1. Random: rank by surrogate reward alone
2. Uncertainty-weighted: rank by reward + ensemble uncertainty bonus

If uncertainty is well-calibrated, the uncertainty-weighted strategy should
find candidates that the oracle ranks higher — it explores the regions where
the surrogate is uncertain and might be wrong.

Run: python -m virtual_lab.domains.fluids.active_demo [--quick]
"""
import argparse
import json
import numpy as np
import time
from ...lab import Lab, copy_state
from .oracle import BurgersOracle
from .benchmarks import load_trained_B
from .models.model_b import SurrogateB
from ...active_loop import random_candidates
from .planner import _rollout_reward
import torch
from pathlib import Path

OUT = Path(__file__).resolve().parent


def load_members():
    """Load 3 ensemble members from trained weights."""
    members = []
    for i in range(3):
        m = SurrogateB(base=32, seed=i)
        wp = OUT / f"weights_C{i}.pt"
        if wp.exists():
            m.net.load_state_dict(
                torch.load(str(wp), map_location="cpu", weights_only=True))
            m.net.eval()
        members.append(m)
    return members


def ensemble_uncertainty(members, state, action, n_steps=5):
    """Compute ensemble variance over a rollout (cheap: just final state)."""
    ends = []
    for m in members:
        sm = copy_state(state)
        for _ in range(n_steps):
            sm = m.predict(sm, action)
        ends.append(np.concatenate([sm["u"].ravel(), sm["v"].ravel()]))
    E = np.stack(ends)
    return float(E.var(0).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-cands", type=int, default=200)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    NC = 50 if a.quick else a.n_cands
    K = 5 if a.quick else a.topk
    IT = 2 if a.quick else a.iters

    B, ok = load_trained_B()
    assert ok, "Need weights_B.pt"
    members = load_members()
    # Use B as the Lab surrogate (Lab expects predict(state,action)->state)
    lab = Lab(BurgersOracle(seed=a.seed + 300), surrogate=B)
    s0 = lab.reset(seed=a.seed, nu=0.01)

    rng = np.random.default_rng(a.seed + 1)
    log = []

    for it in range(IT):
        cands = random_candidates(rng, NC)

        # Strategy 1: random ranking (surrogate reward only)
        rand_scores = []
        for a_cand in cands:
            r = _rollout_reward(lab, s0, [a_cand] * 5, surrogate=True)
            rand_scores.append((r, a_cand))
        rand_scores.sort(key=lambda t: t[0], reverse=True)
        rand_top = [c for _, c in rand_scores[:K]]

        # Strategy 2: uncertainty-weighted (reward + uncertainty bonus)
        unc_scores = []
        for a_cand in cands:
            r = _rollout_reward(lab, s0, [a_cand] * 5, surrogate=True)
            u = ensemble_uncertainty(members, s0, a_cand, n_steps=5)
            unc_scores.append((r + 0.5 * u, r, u, a_cand))
        unc_scores.sort(key=lambda t: t[0], reverse=True)
        unc_top = [c for _, _, _, c in unc_scores[:K]]

        # Oracle validation
        rand_oracle = [_rollout_reward(lab, s0, [c] * 5, surrogate=False)
                       for c in rand_top]
        unc_oracle = [_rollout_reward(lab, s0, [c] * 5, surrogate=False)
                      for c in unc_top]

        # 200-candidate oracle baseline (brute-force truth)
        oracle_all = [_rollout_reward(lab, s0, [c] * 5, surrogate=False)
                      for c in cands]

        row = {
            "iter": it,
            "rand_oracle_mean": float(np.mean(rand_oracle)),
            "rand_oracle_best": float(max(rand_oracle)),
            "unc_oracle_mean": float(np.mean(unc_oracle)),
            "unc_oracle_best": float(max(unc_oracle)),
            "oracle_bruteforce_best": float(max(oracle_all)),
            "unc_mean_var": float(np.mean([u for _, _, u, _ in unc_scores[:K]])),
        }
        log.append(row)
        print(f"iter {it}: rand_oracle={row['rand_oracle_mean']:.4e} "
              f"unc_oracle={row['unc_oracle_mean']:.4e} "
              f"brute_best={row['oracle_bruteforce_best']:.4e} "
              f"unc_var={row['unc_mean_var']:.2e}")

    out = {"iters": IT, "n_cands": NC, "topk": K, "log": log,
           "rand_mean_oracle": float(np.mean([r["rand_oracle_mean"] for r in log])),
           "unc_mean_oracle": float(np.mean([r["unc_oracle_mean"] for r in log]))}
    print(json.dumps({k: v for k, v in out.items() if k != "log"}, indent=2))


if __name__ == "__main__":
    main()
