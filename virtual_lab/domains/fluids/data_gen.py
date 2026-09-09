"""Interventional data gen: random do(A) sequences -> .npz (Y_t, A_t, Y_{t+1}, C)."""
import argparse
import numpy as np
from .oracle import BurgersOracle


def random_action(rng, obstacle_p=0.0, n=64):
    f = None
    if rng.random() < 0.7:
        f = {"amp": float(rng.uniform(-2, 2)), "angle": float(rng.uniform(0, 2 * np.pi)),
             "x": float(rng.random()), "y": float(rng.random()),
             "sigma": float(rng.uniform(0.03, 0.15))}
    ob = None
    if rng.random() < obstacle_p:
        ob = np.zeros((n, n), bool)
        ob[int(n * 0.3):int(n * 0.5), int(n * 0.4):int(n * 0.45)] = True
    return {"d_nu": float(rng.uniform(-0.005, 0.005)), "force": f, "obstacle": ob}


def action_to_vec(a):
    f = a.get("force")
    if f is None:
        return [a.get("d_nu", 0.0), 0, 0, 0.5, 0.5, 0.08]
    return [a.get("d_nu", 0.0), f["amp"], f["angle"], f["x"], f["y"], f["sigma"]]


def generate(trajs=32, steps=50, seed=0, obstacle_p=0.0, n=64):
    rng = np.random.default_rng(seed)
    Yt, At, Yn, C = [], [], [], []
    for tr in range(trajs):
        orc = BurgersOracle(n=n, seed=seed + tr)
        orc.reset(nu=float(rng.uniform(0.001, 0.05)))
        for _ in range(steps):
            s = orc.get_state()
            a = random_action(rng, obstacle_p, n)
            s2 = orc.step(a)
            Yt.append(np.stack([s["u"], s["v"]]))
            At.append(action_to_vec(a))
            Yn.append(np.stack([s2["u"], s2["v"]]))
            C.append(s["C"]["nu"])
    return {"Y_t": np.array(Yt), "A_t": np.array(At),
            "Y_next": np.array(Yn), "C": np.array(C)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="virtual_lab/data.npz")
    p.add_argument("--trajs", type=int, default=32)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    kw = {"trajs": 4, "steps": 10} if a.quick else {"trajs": a.trajs, "steps": a.steps}
    d = generate(seed=a.seed, **kw)
    np.savez_compressed(a.out, **d)
    print(f"saved {a.out}: Y_t {d['Y_t'].shape}")


if __name__ == "__main__":
    main()
