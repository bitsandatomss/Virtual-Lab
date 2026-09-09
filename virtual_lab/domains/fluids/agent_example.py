"""Agent demo: surrogate-guided planning (CEM) vs random shooting.

Runs CEM inside the surrogate to find high-reward action sequences,
then validates the top candidates on the oracle.  Also measures a
10k-screen virtual-only run for comparison.

Run: python -m virtual_lab.domains.fluids.agent_example [--quick]
"""
import argparse
import json
import time
import numpy as np
from ...lab import Lab
from .oracle import BurgersOracle
from .benchmarks import load_trained_B
from .planner import cem_plan, random_shoot, _rollout_reward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    H = 3 if a.quick else 5
    K_cem = 10 if a.quick else 50
    E_cem = 2 if a.quick else 5
    N_rnd = 20 if a.quick else 200
    N_screen = 500 if a.quick else 10000

    B, ok = load_trained_B()
    assert ok, "Need weights_B.pt"
    lab = Lab(BurgersOracle(seed=a.seed + 200), surrogate=B)
    s0 = lab.reset(seed=a.seed, nu=0.01)

    # CEM planning in surrogate
    t0 = time.time()
    cem_seq, cem_virt = cem_plan(lab, s0, H, K=K_cem, iters=E_cem, seed=a.seed,
                                  verbose=False)
    cem_wall = time.time() - t0

    # Random shooting in surrogate
    t0 = time.time()
    rnd_seq, rnd_virt = random_shoot(lab, s0, H, N=N_rnd, seed=a.seed)
    rnd_wall = time.time() - t0

    # 10k virtual screen (random)
    rng = np.random.default_rng(a.seed + 1)
    from ...active_loop import random_candidates
    cands = random_candidates(rng, N_screen)
    t0 = time.time()
    virt_scores = np.array([_rollout_reward(lab, s0, [c] * H, surrogate=True)
                            for c in cands])
    screen_wall = time.time() - t0
    order = np.argsort(virt_scores)[::-1]
    top = [cands[i] for i in order[:10]]

    # Oracle validation of top 10 from each method
    lab_v = Lab(BurgersOracle(seed=a.seed + 200), surrogate=None)
    s0v = lab_v.reset(seed=a.seed, nu=0.01)
    cem_oracle = _rollout_reward(lab_v, s0v, cem_seq, surrogate=False)
    rnd_oracle = _rollout_reward(lab_v, s0v, rnd_seq, surrogate=False)
    top_oracle = np.array([_rollout_reward(lab_v, s0v, [c] * H, surrogate=False)
                           for c in top])

    out = {"horizon": H,
           "cem": {"K": K_cem, "E": E_cem, "virt": cem_virt,
                   "oracle": cem_oracle, "wall_s": cem_wall},
           "random_shoot": {"N": N_rnd, "virt": rnd_virt, "oracle": rnd_oracle,
                            "wall_s": rnd_wall},
           "screen_10k": {"N": N_screen, "virt_best": float(virt_scores[order[0]]),
                          "oracle_top10_mean": float(top_oracle.mean()),
                          "wall_s": screen_wall}}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
