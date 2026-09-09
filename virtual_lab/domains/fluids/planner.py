"""CEM planner in the surrogate vs random shooting.

The core demonstration that the surrogate is an environment, not a predictor:
CEM optimizes action sequences in the surrogate's "imagination" — something
that is impossible without a differentiable/evaluable learned environment.

Run: python -m virtual_lab.domains.fluids.planner [--quick] [--horizon 5]
"""
import argparse
import json
import time
import numpy as np
from ...lab import Lab, copy_state
from .oracle import BurgersOracle
from .models.model_b import SurrogateB
from .benchmarks import load_trained_B


def _sample_actions(rng, K, H, amp_range=(-2.0, 2.0)):
    """Generate K action sequences of length H."""
    amps = rng.uniform(*amp_range, (K, H))
    angles = rng.uniform(0, 2 * np.pi, (K, H))
    actions = []
    for k in range(K):
        seq = [{"d_nu": 0.0,
                "force": {"amp": float(amps[k, h]), "angle": float(angles[k, h]),
                          "x": 0.5, "y": 0.5, "sigma": 0.1}}
               for h in range(H)]
        actions.append(seq)
    return actions


def _rollout_reward(lab, s0, actions, surrogate=True):
    """Sum of squared velocity over the trajectory (higher = more vigorous)."""
    r = 0.0
    s = copy_state(s0)
    for a in actions:
        traj = lab.rollout([a], use_surrogate=surrogate, init=s)
        s = traj[-1]
        r += float(np.mean(s["u"] ** 2 + s["v"] ** 2))
    return r


def cem_plan(lab, s0, H, K=50, elite_frac=0.2, iters=5, amp_range=(-2.0, 2.0),
             seed=0, verbose=True):
    """CEM: iteratively refine a Gaussian over action sequences in the surrogate."""
    rng = np.random.default_rng(seed)
    n_elite = max(2, int(K * elite_frac))
    mu_amp = np.zeros(H)
    mu_ang = np.zeros(H)
    std_amp = np.full(H, 1.0)
    std_ang = np.full(H, np.pi)
    best_seq, best_r = None, -np.inf
    for it in range(iters):
        amps = np.clip(rng.normal(mu_amp, std_amp, (K, H)), *amp_range)
        angles = rng.normal(mu_ang, std_ang, (K, H)) % (2 * np.pi)
        scores = np.empty(K)
        for k in range(K):
            seq = [{"d_nu": 0.0,
                    "force": {"amp": float(amps[k, h]), "angle": float(angles[k, h]),
                              "x": 0.5, "y": 0.5, "sigma": 0.1}}
                   for h in range(H)]
            scores[k] = _rollout_reward(lab, s0, seq, surrogate=True)
        elite_idx = np.argsort(scores)[-n_elite:]
        mu_amp = amps[elite_idx].mean(0)
        mu_ang = angles[elite_idx].mean(0)
        std_amp = np.maximum(amps[elite_idx].std(0), 0.05)
        std_ang = np.maximum(angles[elite_idx].std(0), 0.1)
        if scores[elite_idx[-1]] > best_r:
            best_r = scores[elite_idx[-1]]
            best_seq = [{"d_nu": 0.0,
                         "force": {"amp": float(amps[elite_idx[-1], h]),
                                   "angle": float(angles[elite_idx[-1], h]),
                                   "x": 0.5, "y": 0.5, "sigma": 0.1}}
                        for h in range(H)]
        if verbose:
            print(f"  CEM iter {it}: best={best_r:.4e} "
                  f"elite_mean={scores[elite_idx].mean():.4e}", flush=True)
    return best_seq, best_r


def random_shoot(lab, s0, H, N=50, amp_range=(-2.0, 2.0), seed=0):
    """Baseline: sample N random sequences, return the best."""
    rng = np.random.default_rng(seed)
    actions_list = _sample_actions(rng, N, H, amp_range)
    best_seq, best_r = None, -np.inf
    for seq in actions_list:
        r = _rollout_reward(lab, s0, seq, surrogate=True)
        if r > best_r:
            best_r = r
            best_seq = seq
    return best_seq, best_r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--cem-k", type=int, default=50)
    p.add_argument("--cem-iters", type=int, default=5)
    p.add_argument("--random-n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    H = 3 if a.quick else a.horizon
    K = 10 if a.quick else a.cem_k
    RI = 2 if a.quick else a.cem_iters
    RN = 20 if a.quick else a.random_n

    B, ok = load_trained_B()
    assert ok, "Need weights_B.pt"
    lab = Lab(BurgersOracle(seed=a.seed + 200), surrogate=B)
    s0 = lab.reset(seed=a.seed, nu=0.01)

    # CEM
    t0 = time.time()
    cem_seq, cem_r = cem_plan(lab, s0, H, K=K, iters=RI, seed=a.seed)
    cem_time = time.time() - t0
    # Random
    t0 = time.time()
    rnd_seq, rnd_r = random_shoot(lab, s0, H, N=RN, seed=a.seed)
    rnd_time = time.time() - t0
    # Oracle validation (top 3 sequences each)
    lab_v = Lab(BurgersOracle(seed=a.seed + 200), surrogate=None)
    s0v = lab_v.reset(seed=a.seed, nu=0.01)
    cem_oracle = _rollout_reward(lab_v, s0v, cem_seq, surrogate=False)
    rnd_oracle = _rollout_reward(lab_v, s0v, rnd_seq, surrogate=False)

    out = {"horizon": H, "cem": {"K": K, "iters": RI, "virt_reward": cem_r,
                                  "oracle_reward": cem_oracle, "wall_s": cem_time},
           "random": {"N": RN, "virt_reward": rnd_r, "oracle_reward": rnd_oracle,
                      "wall_s": rnd_time},
           "oracle_speedup": rnd_time / max(cem_time, 1e-9)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
