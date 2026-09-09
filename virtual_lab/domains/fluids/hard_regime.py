"""Hard regime evaluation: low-viscosity shocks (nu=0.001).

The current model was trained on nu∈[0.01, 0.02]. This test probes the
boundary: nu=0.001 produces shocks that the smooth training data never saw.
Expected: performance degrades vs in-distribution (nu=0.01), quantifying
exactly how much the surrogate breaks outside its training envelope.

Run: python -m virtual_lab.domains.fluids.hard_regime [--retrain]
"""
import argparse
import json
import numpy as np
import torch
import time
from pathlib import Path

from .oracle import BurgersOracle
from ...lab import Lab, state_error, copy_state
from .models.model_b import SurrogateB

OUT = Path(__file__).resolve().parent


def eval_at_nu(B, nu, n_eval=50, seed=0):
    """Mean rollout error at a given viscosity over n_eval sequences."""
    errs = []
    orc = BurgersOracle(seed=seed + 700)
    for k in range(n_eval):
        s0 = orc.reset(seed=seed + k, nu=nu)
        sm = copy_state(s0)
        for _ in range(10):
            a = {"d_nu": 0.0, "force": {"amp": 1.0, "angle": 0.3 * k,
                                        "x": 0.5, "y": 0.5, "sigma": 0.1}}
            s_true = orc.step(a)
            sm = B.predict(sm, a)
            errs.append(state_error(s_true, sm))
    return float(np.mean(errs))


def retrain_hard():
    """Retrain with a mix of normal + low-nu data."""
    from .train_full import train_member, save_weights
    from .data_gen import generate
    full = dict(np.load(OUT / "data_full.npz"))
    # Generate extra low-nu trajectories (nu∈[0.001, 0.005])
    d_low = generate(trajs=4, steps=25, seed=999)
    Yt = np.concatenate([full["Y_t"], d_low["Y_t"]]).astype(np.float32)
    Yn = np.concatenate([full["Y_next"], d_low["Y_next"]]).astype(np.float32)
    A = np.concatenate([full["A_t"], d_low["A_t"]]).astype(np.float32)
    C = np.concatenate([full["C"], d_low["C"]]).astype(np.float32)
    X = np.stack([Yt[:, 0], Yt[:, 1],
                  C[:, None, None] * 20.0,
                  A[:, 0:1] * 0.5, A[:, 1:2] * 0.25], 1)
    Y = Yn - Yt
    sb = SurrogateB(seed=0)
    loss = train_member(sb, X, Y, epochs=10, batch=64, seed=0, device="cuda",
                        tag="B-hard")
    save_weights(OUT / "weights_B_hard.pt", sb.net)
    return sb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--retrain", action="store_true")
    a = p.parse_args()

    B = SurrogateB(seed=0)
    if a.retrain:
        B = retrain_hard()
        tag = "retrained"
    else:
        wp = OUT / "weights_B.pt"
        if wp.exists():
            B.net.load_state_dict(torch.load(str(wp), map_location="cpu",
                                              weights_only=True))
            B.net.eval()
        tag = "original"

    nus = [0.01, 0.005, 0.002, 0.001]
    results = {}
    for nu in nus:
        t0 = time.time()
        err = eval_at_nu(B, nu, n_eval=20)
        results[f"nu_{nu}"] = {"error": err, "wall_s": time.time() - t0}
        print(f"  nu={nu}: err={err:.2e} ({results[f'nu_{nu}']['wall_s']:.1f}s)",
              flush=True)

    out = {"model": tag, "results": results}
    (OUT / "HARD_REGIME.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
