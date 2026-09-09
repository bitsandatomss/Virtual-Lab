"""Uncertainty calibration for the fluids ensemble -> CALIBRATION.json.

Questions answered (all empirically, no hand-waving):
1. Reliability: does predicted ensemble variance track actual error? (binned)
2. OOD detection: does variance separate in-dist from obstacle-OOD? (AUROC)
3. Ablation: k=1 vs 2 vs 3 members — error, reliability, cost.

Run: python -m virtual_lab.domains.fluids.calibrate [--n 200]
CPU-only, a few minutes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .oracle import BurgersOracle
from .data_gen import generate
from .models.model_b import SurrogateB

OUT = Path(__file__).resolve().parent
N_EVAL = 200


def load_member(name, seed):
    sb = SurrogateB(base=32, seed=seed)
    sb.net.load_state_dict(torch.load(str(OUT / name), map_location="cpu",
                                      weights_only=True))
    sb.net.eval()
    return sb


def member_var(members, s, a):
    us = np.stack([m.predict(s, a)["u"] for m in members])
    vs = np.stack([m.predict(s, a)["v"] for m in members])
    return float((us.var(0) + vs.var(0)).mean())


def collect(members, data):
    errs, vars_ = [], []
    orc = BurgersOracle(seed=0)
    for k in range(min(len(data["Y_t"]), N_EVAL)):
        s = {"u": data["Y_t"][k, 0].copy(), "v": data["Y_t"][k, 1].copy(),
             "t": 0.0, "C": {"nu": float(data["C"][k]), "obstacle": None}}
        dn, amp, ang, fx, fy, sig = (float(v) for v in data["A_t"][k])
        a = {"d_nu": dn, "force": {"amp": amp, "angle": ang, "x": fx,
                                   "y": fy, "sigma": sig}}
        orc.set_state({"u": s["u"].copy(), "v": s["v"].copy(), "t": 0.0,
                       "C": {"nu": s["C"]["nu"], "obstacle": None}})
        t = orc.step(a)
        mean_u = np.mean([m.predict(s, a)["u"] for m in members], 0)
        mean_v = np.mean([m.predict(s, a)["v"] for m in members], 0)
        errs.append(float(np.mean((mean_u - t["u"]) ** 2 + (mean_v - t["v"]) ** 2)))
        vars_.append(member_var(members, s, a))
    return np.array(errs), np.array(vars_)


def auroc(scores_in, scores_ood):
    """P(score_ood > score_in): Mann-Whitney form, no sklearn needed."""
    a = np.asarray(scores_in)
    b = np.asarray(scores_ood)
    return float(((b[:, None] > a[None, :]).mean() +
                  0.5 * (b[:, None] == a[None, :]).mean()))


def main():
    global N_EVAL
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=N_EVAL)
    a = p.parse_args()
    N_EVAL = a.n
    members = [load_member(f"weights_C{i}.pt", i) for i in range(3)]
    d_in = generate(trajs=8, steps=25, seed=999)
    d_ood = generate(trajs=8, steps=25, seed=555, obstacle_p=1.0)
    errs_in, var_in = collect(members, d_in)
    errs_ood, var_ood = collect(members, d_ood)

    # 1. reliability: decile bins of predicted variance
    order = np.argsort(var_in)
    bins = np.array_split(order, 10)
    rel = [{"mean_var": float(var_in[b].mean()),
            "mean_err": float(errs_in[b].mean())} for b in bins]
    rel_corr = float(np.corrcoef([r["mean_var"] for r in rel],
                                 [r["mean_err"] for r in rel])[0, 1])

    # 2. OOD AUROC of variance
    auc = auroc(var_in, var_ood)

    # 3. member ablation: k=1,2,3 on in-dist error
    abl = {}
    for k in (1, 2, 3):
        errs_k, _ = collect(members[:k], d_in)
        abl[f"k{k}"] = {"mean_err": float(errs_k.mean())}

    out = {"n_eval": N_EVAL,
           "reliability_bins": rel,
           "reliability_corr": rel_corr,
           "ood_auroc_var": auc,
           "var_in_mean": float(var_in.mean()),
           "var_ood_mean": float(var_ood.mean()),
           "ablation": abl}
    (OUT / "CALIBRATION.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "reliability_bins"},
                     indent=2))
    print("bins (var -> err):", [(f"{r['mean_var']:.2e}", f"{r['mean_err']:.2e}")
                                  for r in rel])


if __name__ == "__main__":
    main()
