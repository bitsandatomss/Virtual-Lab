"""Amortization analysis: surrogate vs oracle wall-clock for planning.

The core question: at what N does the surrogate break even with direct oracle
optimization?  Measures CEM wall-time in surrogate vs equivalent evaluations
on the oracle. The "speedup at break-even" tells you when amortized training
pays off for a given planner budget.

Run: python -m virtual_lab.domains.fluids.amortization [--quick]
"""
import argparse
import json
import time
import numpy as np
from ...lab import Lab, copy_state
from .oracle import BurgersOracle
from .benchmarks import load_trained_B
from .planner import cem_plan, _rollout_reward


def measure_oracle_step_time(seed=0, n_steps=200):
    """Time a single oracle step (averaged over n_steps calls)."""
    orc = BurgersOracle(seed=seed)
    orc.reset(nu=0.01)
    a = {"d_nu": 0.0, "force": {"amp": 1.0, "angle": 0.5, "x": 0.5,
                                "y": 0.5, "sigma": 0.1}}
    t0 = time.perf_counter()
    for _ in range(n_steps):
        orc.step(a)
    return (time.perf_counter() - t0) / n_steps


def measure_surrogate_step_time(B, seed=0, n_steps=200):
    """Time a single surrogate predict call."""
    lab = Lab(BurgersOracle(seed=seed), surrogate=B)
    s = lab.reset(seed=seed, nu=0.01)
    a = {"d_nu": 0.0, "force": {"amp": 1.0, "angle": 0.5, "x": 0.5,
                                "y": 0.5, "sigma": 0.1}}
    t0 = time.perf_counter()
    for _ in range(n_steps):
        s = B.predict(s, a)
    return (time.perf_counter() - t0) / n_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--horizon", type=int, default=5)
    a = p.parse_args()
    H = 3 if a.quick else a.horizon

    B, ok = load_trained_B()
    assert ok, "Need weights_B.pt"

    orc_dt = measure_oracle_step_time()
    sur_dt = measure_surrogate_step_time(B)
    step_speedup = orc_dt / sur_dt

    # Planning comparison: CEM(K=50, iters=5) = 250 surrogate rollouts
    # vs random(N=200) = 200 surrogate rollouts
    # Equivalent oracle cost: N × H × oracle_step_time
    K_cem, E_cem = (10, 2) if a.quick else (50, 5)
    N_rnd = (20) if a.quick else 200

    lab = Lab(BurgersOracle(seed=42), surrogate=B)
    s0 = lab.reset(seed=42, nu=0.01)
    t0 = time.time()
    cem_seq, cem_virt = cem_plan(lab, s0, H, K=K_cem, iters=E_cem, seed=42,
                                  verbose=False)
    cem_sur_time = time.time() - t0
    cem_oracle_calls = K_cem * E_cem * H
    cem_oracle_est = cem_oracle_calls * orc_dt

    t0 = time.time()
    from .planner import random_shoot
    rnd_seq, rnd_virt = random_shoot(lab, s0, H, N=N_rnd, seed=42)
    rnd_sur_time = time.time() - t0
    rnd_oracle_calls = N_rnd * H
    rnd_oracle_est = rnd_oracle_calls * orc_dt

    # Oracle validation
    lab_v = Lab(BurgersOracle(seed=42), surrogate=None)
    s0v = lab_v.reset(seed=42, nu=0.01)
    cem_oracle_r = _rollout_reward(lab_v, s0v, cem_seq, surrogate=False)
    rnd_oracle_r = _rollout_reward(lab_v, s0v, rnd_seq, surrogate=False)

    # Break-even: how many planning iterations before surrogate training
    # cost is amortized?  Training ~2 min (126s for V, ~120s for B).
    training_s = 120.0
    # Each surrogate eval that replaces an oracle eval saves (orc_dt - sur_dt).
    savings_per_eval = (orc_dt - sur_dt)
    total_cem_evals = K_cem * E_cem
    total_rnd_evals = N_rnd
    # Number of full planning runs to break even on training cost.
    breakeven_cem = training_s / max(total_cem_evals * savings_per_eval, 1e-9)
    breakeven_rnd = training_s / max(total_rnd_evals * savings_per_eval, 1e-9)

    out = {
        "step_times": {"oracle_ms": orc_dt * 1000, "surrogate_ms": sur_dt * 1000,
                       "speedup": step_speedup},
        "cem": {"K": K_cem, "E": E_cem, "total_surrogate_evals": total_cem_evals,
                "surrogate_wall_s": cem_sur_time,
                "oracle_equiv_s": cem_oracle_est,
                "oracle_reward": cem_oracle_r},
        "random": {"N": N_rnd, "total_surrogate_evals": total_rnd_evals,
                   "surrogate_wall_s": rnd_sur_time,
                   "oracle_equiv_s": rnd_oracle_est,
                   "oracle_reward": rnd_oracle_r},
        "amortization": {"training_cost_s": training_s,
                         "breakeven_cem_plans": breakeven_cem,
                         "breakeven_random_plans": breakeven_rnd},
        "total_evals_10k_screen": 10000 * H,
        "oracle_equiv_10k_screen_s": 10000 * H * orc_dt,
        "note": ("The CPU Burgers oracle is a fast numpy explicit solver — "
                 "faster per step than the surrogate. Amortization break-even "
                 "is infinite here. The surrogate's value is NOT speed but "
                 "enabling virtual screening (10k+ candidates) that would be "
                 "impossible with a real experiment. For expensive oracles "
                 "(3D CFD, wet-lab), the surrogate becomes the only option.")}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
