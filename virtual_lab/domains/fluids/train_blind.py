"""Train Model V (blind predictor): same data/epochs/optimizer as Model B,
3-channel [u, v, nu] inputs, no action channels anywhere.

Run: python -m virtual_lab.domains.fluids.train_blind [--device auto|cpu|cuda]
"""
import argparse
import time
import types
from pathlib import Path

import numpy as np
import torch

from .train_full import train_member, save_weights
from .models.model_b import TinyUNet

OUT = Path(__file__).resolve().parent


def build_x_blind(data, n=64):
    Yt = np.asarray(data["Y_t"], dtype=np.float32)
    C = np.asarray(data["C"], dtype=np.float32)
    X = np.empty((len(Yt), 3, n, n), dtype=np.float32)
    X[:, 0], X[:, 1] = Yt[:, 0], Yt[:, 1]
    X[:, 2] = C[:, None, None] * 20.0
    Y = np.asarray(data["Y_next"], dtype=np.float32) - Yt
    return X, Y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    a = p.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    print(f"device={dev}", flush=True)
    full = dict(np.load(OUT / "data_full.npz"))
    X, Y = build_x_blind(full)
    print(f"X{X.shape} Y{Y.shape}", flush=True)
    sb = types.SimpleNamespace(net=TinyUNet(a.base, in_ch=3), opt=None)
    t0 = time.time()
    loss = train_member(sb, X, Y, a.epochs, a.batch, a.seed, dev, tag="V")
    save_weights(OUT / "weights_V.pt", sb.net)
    print(f"V final loss {loss:.6e} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
