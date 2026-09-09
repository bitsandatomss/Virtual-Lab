"""Full training for the Virtual Lab Learned Environment (reproducible).

Steps 1-3 of the full-training task:
  1. Generate full dataset: generate(trajs=200, steps=50, seed=0) (~10k pairs)
     -> data_full.npz, plus holdout (seed=123) and OOD sets
     (obstacle_p=1.0, fixed nu=0.05) for evaluation. n=64.
  2. Train Model B (SurrogateB, base=32) for epochs=25, batch=64 -> weights_B.pt.
  3. Train Model C: 3x SurrogateB seeds 0,1,2, epochs=15 each -> weights_C{0,1,2}.pt.

Run: python -m virtual_lab.domains.fluids.train_full [--device auto|cpu|cuda] [--reuse-data]

Speed note: SurrogateB.fit() re-rasterizes every force field from scratch each
epoch on CPU (~80+ min for B alone at this size, ~4h total with C). This script
therefore precomputes the EXACT same input features fit() builds
[u, v, fx, fy, nu*20, zeros-obstacle] and residual targets (Y_next - Y_t),
uses the same TinyUNet(base=32) / Adam(lr=1e-3) / MSELoss / batch=64 /
epochs, and saves plain net.state_dict() files loadable into SurrogateB.
Device defaults to CUDA when available (CPU fallback) to meet the ~10-min
budget; the math, features, loss, optimizer, batch size and epoch counts are
unchanged. Field arrays are stored as float32 (shapes identical).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data_gen import generate, random_action, action_to_vec
from .oracle import BurgersOracle
from .models.model_b import SurrogateB

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
print(f"OUT={OUT}", flush=True)


def gen_nu_fixed(trajs=16, steps=20, seed=21, nu=0.05, obstacle_p=0.0, n=64):
    """Same pipeline as data_gen.generate but with fixed viscosity nu."""
    rng = np.random.default_rng(seed)
    Yt, At, Yn, C = [], [], [], []
    for tr in range(trajs):
        orc = BurgersOracle(n=n, seed=seed + tr)
        orc.reset(nu=nu)
        for _ in range(steps):
            s = orc.get_state()
            a = random_action(rng, obstacle_p, n)
            a["d_nu"] = 0.0  # keep nu fixed at the OOD value
            s2 = orc.step(a)
            Yt.append(np.stack([s["u"], s["v"]]))
            At.append(action_to_vec(a))
            Yn.append(np.stack([s2["u"], s2["v"]]))
            C.append(s["C"]["nu"])
    return {"Y_t": np.array(Yt), "A_t": np.array(At),
            "Y_next": np.array(Yn), "C": np.array(C)}


def to_f32(d):
    d = dict(d)
    d["Y_t"] = np.asarray(d["Y_t"], dtype=np.float32)
    d["Y_next"] = np.asarray(d["Y_next"], dtype=np.float32)
    return d


def save_npz(path, d):
    np.savez_compressed(os.fspath(path), **d)
    assert os.path.exists(path) and os.path.getsize(path) > 0, f"save failed: {path}"
    print(f"saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)", flush=True)


def save_weights(path, net):
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()},
               os.fspath(path))
    assert os.path.exists(path) and os.path.getsize(path) > 0, f"save failed: {path}"
    print(f"saved {path} ({os.path.getsize(path) / 1e3:.1f} kB)", flush=True)


def build_xy(data, n=64):
    """Precompute EXACTLY what SurrogateB.fit() builds per batch.

    fit() per sample k: force dict from A_t row -> rasterize_force,
    x = [u, v, fx, fy, nu*20, zeros]; target = Y_next - Y_t (residual).
    (Note: fit() uses a zeros obstacle channel, replicated here.)
    """
    from .oracle import rasterize_force
    Yt, Yn, A, C = data["Y_t"], data["Y_next"], data["A_t"], data["C"]
    N = len(Yt)
    X = np.empty((N, 6, n, n), dtype=np.float32)
    for k in range(N):
        d_nu, amp, ang, fx_, fy_, sig = A[k]
        force = {"amp": float(amp), "angle": float(ang),
                 "x": float(fx_), "y": float(fy_), "sigma": float(sig)}
        fxr, fyr = rasterize_force(force, n)
        X[k, 0] = Yt[k, 0]
        X[k, 1] = Yt[k, 1]
        X[k, 2] = fxr
        X[k, 3] = fyr
        X[k, 4] = C[k] * 20.0
        X[k, 5] = 0.0
    Y = (np.asarray(Yn, dtype=np.float32) - np.asarray(Yt, dtype=np.float32))
    return X, Y


def train_member(sb, X, Y, epochs, batch, seed, device, tag):
    """Minibatch Adam training on precomputed features (same hyperparams as fit)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    sb.net.to(device)
    sb.opt = torch.optim.Adam(sb.net.parameters(), lr=1e-3)  # same as __init__
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)
    N = len(X)
    last = None
    for ep in range(epochs):
        idx = np.arange(N)
        rng.shuffle(idx)
        tot, nb = 0.0, 0
        sb.net.train()
        for i in range(0, N, batch):
            bi = idx[i:i + batch]
            xb, yb = Xt[bi], Yt[bi]
            sb.opt.zero_grad()
            loss = loss_fn(sb.net(xb), yb)
            loss.backward()
            sb.opt.step()
            tot += float(loss.item())
            nb += 1
        last = tot / nb
        print(f"[{tag}] epoch {ep + 1}/{epochs} mean-batch-loss {last:.6e}", flush=True)
    return last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajs", type=int, default=200)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs-b", type=int, default=25)
    p.add_argument("--epochs-c", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--reuse-data", action="store_true",
                   help="load existing npz files instead of regenerating")
    a = p.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    print(f"device={dev} torch={torch.__version__}", flush=True)
    t_all = time.time()
    log = {"config": vars(a), "device": dev, "out": str(OUT)}

    # ---- 1. datasets ----
    paths = {"full": OUT / "data_full.npz", "holdout": OUT / "data_holdout.npz",
             "ood_obstacle": OUT / "data_ood_obstacle.npz",
             "ood_nu": OUT / "data_ood_nu.npz"}
    if a.reuse_data and all(os.path.exists(v) for v in paths.values()):
        t0 = time.time()
        full = dict(np.load(paths["full"]))
        hold = dict(np.load(paths["holdout"]))
        ood_ob = dict(np.load(paths["ood_obstacle"]))
        ood_nu = dict(np.load(paths["ood_nu"]))
        print(f"reused data ({time.time()-t0:.1f}s)", flush=True)
    else:
        t0 = time.time()
        full = to_f32(generate(trajs=a.trajs, steps=a.steps, seed=a.seed))
        save_npz(paths["full"], full)
        hold = to_f32(generate(trajs=32, steps=50, seed=123))
        save_npz(paths["holdout"], hold)
        ood_ob = to_f32(generate(trajs=16, steps=20, seed=11, obstacle_p=1.0))
        save_npz(paths["ood_obstacle"], ood_ob)
        ood_nu = to_f32(gen_nu_fixed(trajs=16, steps=20, seed=21, nu=0.05))
        save_npz(paths["ood_nu"], ood_nu)
        print(f"data generation took {time.time()-t0:.1f}s", flush=True)
    log["dataset"] = {k: {"shape": [list(map(int, np.asarray(v).shape))
                                    for v in d.values()], "keys": list(d.keys())}
                      for k, d in [("full", full), ("holdout", hold),
                                   ("ood_obstacle", ood_ob), ("ood_nu", ood_nu)]}
    log["dataset_shapes"] = {key: list(map(int, np.asarray(full[key]).shape))
                             for key in full}
    print(f"data_full: Y_t {np.asarray(full['Y_t']).shape}", flush=True)
    print(f"holdout {np.asarray(hold['Y_t']).shape}, "
          f"ood_ob {np.asarray(ood_ob['Y_t']).shape}, "
          f"ood_nu {np.asarray(ood_nu['Y_t']).shape}", flush=True)

    # ---- shared precomputed features (identical to fit()) ----
    t0 = time.time()
    X, Y = build_xy(full)
    print(f"precompute X{X.shape} Y{Y.shape} ({time.time()-t0:.1f}s)", flush=True)

    # ---- 2. Model B ----
    t0 = time.time()
    sb = SurrogateB(base=a.base, seed=0)
    loss_b = train_member(sb, X, Y, a.epochs_b, a.batch, seed=0, device=dev, tag="B")
    save_weights(OUT / "weights_B.pt", sb.net)
    log["train_B"] = {"epochs": a.epochs_b, "batch": a.batch, "base": a.base,
                      "seed": 0, "final_loss": loss_b, "time_s": time.time() - t0}
    print(f"B final loss {loss_b:.6e}", flush=True)

    # ---- 3. Model C: 3 members, seeds 0,1,2 ----
    log["train_C"] = []
    for s in (0, 1, 2):
        t0 = time.time()
        m = SurrogateB(base=a.base, seed=s)
        loss = train_member(m, X, Y, a.epochs_c, a.batch, seed=100 + s,
                            device=dev, tag=f"C{s}")
        save_weights(OUT / f"weights_C{s}.pt", m.net)
        log["train_C"].append({"epochs": a.epochs_c, "batch": a.batch,
                               "base": a.base, "seed": s, "final_loss": loss,
                               "time_s": time.time() - t0})
        print(f"C{s} final loss {loss:.6e}", flush=True)

    log["total_time_s"] = time.time() - t_all
    with open(os.fspath(OUT / "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))
    print("OUT contents:", sorted(os.path.basename(v) for v in OUT.glob("*")
                                  if os.path.isfile(v)), flush=True)


if __name__ == "__main__":
    main()
