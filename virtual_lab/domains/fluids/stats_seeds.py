"""Multi-seed benchmark statistics -> SEEDS.json (mean/std/95% CI).

Run: python -m virtual_lab.domains.fluids.stats_seeds [--seeds 0 1 2 3 4]
Full (non-quick) protocol per seed. Slow (~minutes/seed, CPU surrogate
inference dominates); results quantify sampling variance behind RESULTS.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .benchmarks import run_all

OUT = Path(__file__).resolve().parent / "SEEDS.json"
KEYS = ["T1_one_step_B", "T2_rollout10_B", "T2_rollout10_A", "T3_indist_holdout_B",
        "T3_shift_A", "T3_shift_V", "T3_shift_B", "T4_obstacle_B", "T5_nu_extrap_B",
        "T6_H_eps1e-3", "T6_decision_H", "T7_R_discovery_top5", "T7_Delta_V2R",
        "T8_corr_A", "T8_corr_V", "T8_corr_B", "T8_sens_A", "T8_sens_V", "T8_sens_B"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = p.parse_args()
    rows = []
    for s in a.seeds:
        print(f"seed {s} ...", flush=True)
        T = run_all(quick=False, seed=s)
        rows.append({k: T.get(k) for k in KEYS})
        print(f"  T1={T['T1_one_step_B']:.2e} T8corrB={T['T8_corr_B']:.3f} "
              f"T7R={T['T7_R_discovery_top5']:.2f}", flush=True)
    stats = {}
    for k in KEYS:
        vals = np.array([r[k] for r in rows if r[k] is not None], dtype=float)
        stats[k] = {"n": int(len(vals)), "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "ci95": float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))
                    if len(vals) > 1 else 0.0}
    out = {"seeds": a.seeds, "rows": rows, "stats": stats}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
