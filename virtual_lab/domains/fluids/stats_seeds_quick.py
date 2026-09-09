"""Quick-mode multi-seed stats -> SEEDS.json. 5 seeds, ~30s each."""
import json
import numpy as np
from pathlib import Path
from virtual_lab.domains.fluids.benchmarks import run_all

OUT = Path(__file__).resolve().parent / "SEEDS.json"
KEYS = ["T1_one_step_B", "T2_rollout10_B", "T8_corr_B", "T8_sens_B",
        "T7_R_discovery_top5", "T6_decision_H", "T3_shift_B"]

def main():
    rows = []
    for s in range(5):
        T = run_all(quick=True, seed=s)
        rows.append({k: T.get(k) for k in KEYS})
        print(f"seed {s}: T1={T['T1_one_step_B']:.2e} T8corr={T['T8_corr_B']:.3f} "
              f"T7R={T['T7_R_discovery_top5']:.2f} T6dH={T['T6_decision_H']}")
    stats = {}
    for k in KEYS:
        vals = np.array([r[k] for r in rows if r[k] is not None], dtype=float)
        stats[k] = {"n": int(len(vals)), "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "ci95": float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))
                    if len(vals) > 1 else 0.0}
    out = {"seeds": list(range(5)), "rows": rows, "stats": stats}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
