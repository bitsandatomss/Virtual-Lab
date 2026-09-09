"""Benchmarks Tests 1-7 (fluids domain). Run: python -m virtual_lab.domains.fluids.benchmarks --quick"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from .oracle import BurgersOracle
from ...lab import Lab, state_error, copy_state
from .data_gen import generate
from .models.model_a import PersistenceBaseline
from .models.model_b import SurrogateB
from .models.model_blind import BlindPredictor

WEIGHTS_B = Path(__file__).resolve().parent / "weights_B.pt"
WEIGHTS_V = Path(__file__).resolve().parent / "weights_V.pt"


def load_trained_B():
    """Load the fully-trained Model B weights.

    Returns (B, True). If weights are missing, returns an untrained net with
    (B, False) — callers must train a fallback AND flag the numbers as
    non-comparable to RESULTS.json. Never silently evaluate an untrained net.
    """
    B = SurrogateB()
    if WEIGHTS_B.exists():
        B.net.load_state_dict(torch.load(str(WEIGHTS_B), map_location="cpu",
                                         weights_only=True))
        B.net.eval()
        return B, True
    print("WARNING: weights_B.pt not found — falling back to a freshly trained "
          "small net. Numbers will NOT match RESULTS.json. Run train_full.py.")
    return B, False


def load_trained_V():
    """Load the fully-trained blind predictor (action-ablation control)."""
    V = BlindPredictor()
    if WEIGHTS_V.exists():
        V.net.load_state_dict(torch.load(str(WEIGHTS_V), map_location="cpu",
                                         weights_only=True))
        V.net.eval()
        return V, True
    print("WARNING: weights_V.pt not found — T8 blind ablation skipped. "
          "Run train_blind.py.")
    return None, False


def _pairwise_dists(ends):
    E = np.stack([np.concatenate([s["u"].ravel(), s["v"].ravel()]) for s in ends])
    d = ((E[:, None, :] - E[None, :, :]) ** 2).mean(-1)
    iu = np.triu_indices(len(E), 1)
    return d[iu]


def _roll_err(model, seed=0, steps=20, nu=0.01):
    lab = Lab(BurgersOracle(seed=seed), surrogate=None)
    s0 = lab.reset(seed=seed, nu=nu)
    errs, s = [], dict(s0)
    orc = BurgersOracle(seed=seed + 999)
    orc.set_state(lab.branch(s0))
    sm = dict(s0)
    for i in range(steps):
        a = {"d_nu": 0.0, "force": {"amp": 1.0, "angle": 0.3 * i,
                                    "x": 0.5, "y": 0.5, "sigma": 0.1}}
        s_true = orc.step(a)
        sm = model.predict(sm, a) if hasattr(model, "predict") else model.predict_mean(sm, a)
        errs.append(state_error(s_true, sm))
    return errs


def run_all(quick=False, seed=0, retrain=False):
    rng = np.random.default_rng(seed)
    steps = 5 if quick else 20
    A = PersistenceBaseline()
    B, from_weights = (None, False) if retrain else load_trained_B()
    if B is None:
        B = SurrogateB()
    # Eval data is always freshly generated (never the training set).
    data = generate(trajs=4 if quick else 16, steps=10 if quick else 30, seed=seed)
    if not from_weights:
        B.fit(data, epochs=1 if quick else 3)
    V, v_ok = load_trained_V()

    def mse1(model, train_like=True):
        d = data if train_like else generate(trajs=4, steps=10, seed=seed + 7)
        e = []
        for k in range(min(len(d["Y_t"]), 50 if quick else 200)):
            s = {"u": d["Y_t"][k, 0], "v": d["Y_t"][k, 1], "t": 0.0, "C": {"nu": float(d["C"][k])}}
            dn, amp, ang, fx, fy, sig = d["A_t"][k]
            a = {"d_nu": float(dn), "force": {"amp": float(amp), "angle": float(ang),
                                              "x": float(fx), "y": float(fy), "sigma": float(sig)}}
            t = {"u": d["Y_next"][k, 0], "v": d["Y_next"][k, 1], "t": 0.01, "C": {"nu": 0.01}}
            p = model.predict(s, a)
            e.append(state_error(t, p))
        return float(np.mean(e))

    T = {}
    T["T1_one_step_B"] = mse1(B, True)
    T["T2_rollout10_B"] = float(np.mean(_roll_err(B, steps=10 if quick else 20)))
    T["T2_rollout10_A"] = float(np.mean(_roll_err(A, steps=10 if quick else 20)))
    T["T3_indist_holdout_B"] = mse1(B, False)
    # T3b amplitude shift: amp=3.0 lies outside the training range [-2, 2].
    # Honest interventional shift (the old "holdout" above is in-distribution).
    d7 = generate(trajs=4, steps=10, seed=seed + 7)
    orcS = BurgersOracle(seed=seed + 777)
    for name, model in [("A", A), ("V", V), ("B", B)]:
        if model is None:
            T[f"T3_shift_{name}"] = None
            continue
        e = []
        for k in range(len(d7["Y_t"])):
            s = {"u": d7["Y_t"][k, 0].copy(), "v": d7["Y_t"][k, 1].copy(),
                 "t": 0.0, "C": {"nu": float(d7["C"][k]), "obstacle": None}}
            _dn, _a0, ang, fx, fy, sig = (float(v) for v in d7["A_t"][k])
            a = {"d_nu": 0.0, "force": {"amp": 3.0, "angle": ang, "x": fx,
                                        "y": fy, "sigma": sig}}
            orcS.set_state({"u": s["u"].copy(), "v": s["v"].copy(), "t": 0.0,
                            "C": {"nu": s["C"]["nu"], "obstacle": None}})
            tstate = orcS.step(a)
            e.append(state_error(tstate, model.predict(s, a)))
        T[f"T3_shift_{name}"] = float(np.mean(e))
    # T4 obstacle OOD
    d_ob = generate(trajs=2, steps=5, seed=seed + 11, obstacle_p=1.0)
    T["T4_obstacle_B"] = mse1(B, True) and float(np.mean(
        [state_error({"u": d_ob["Y_next"][k, 0], "v": d_ob["Y_next"][k, 1], "t": 0, "C": {"nu": 0.01}},
                     B.predict({"u": d_ob["Y_t"][k, 0], "v": d_ob["Y_t"][k, 1], "t": 0,
                                    "C": {"nu": float(d_ob["C"][k])}},
                               {"d_nu": float(d_ob["A_t"][k, 0]),
                                "force": {"amp": float(d_ob["A_t"][k, 1]),
                                          "angle": float(d_ob["A_t"][k, 2]),
                                          "x": float(d_ob["A_t"][k, 3]),
                                          "y": float(d_ob["A_t"][k, 4]),
                                          "sigma": float(d_ob["A_t"][k, 5])}}))
         for k in range(len(d_ob["Y_t"]))]))
    # T5 nu extrapolation (high viscosity)
    T["T5_nu_extrap_B"] = float(np.mean(_roll_err(B, seed=3, nu=0.05, steps=steps)))
    # T6 validated horizon
    lab = Lab(BurgersOracle(), surrogate=B)
    s0 = lab.reset(seed=seed, nu=0.01)
    acts = [{"d_nu": 0.0, "force": {"amp": 1.0, "angle": 0.2 * i,
                                    "x": 0.5, "y": 0.5, "sigma": 0.1}} for i in range(20)]
    T["T6_H_eps1e-3"] = lab.validated_horizon(acts, eps=1e-3, init=s0)
    # T6b decision horizon: first rollout length where the surrogate's top-1
    # candidate differs from the oracle's. Decision-grounded, not MSE-based.
    acts6 = [{"d_nu": 0.0, "force": {"amp": a, "angle": g, "x": 0.5, "y": 0.5,
                                     "sigma": 0.1}}
             for a, g in [(1.5, 0.0), (-1.5, 0.0), (1.5, np.pi / 2),
                          (-1.5, np.pi / 2), (0.2, 0.0), (2.0, np.pi)]]
    labD = Lab(BurgersOracle(seed=seed + 889), surrogate=B)
    s0d = labD.reset(seed=seed, nu=0.01)

    def _e(roll):
        u, v = roll[-1]["u"], roll[-1]["v"]
        return float(np.mean(u ** 2 + v ** 2))

    T["T6_decision_H"] = 16
    for h in range(1, 16):
        vv = [_e(labD.rollout([a] * h, use_surrogate=True, init=s0d)) for a in acts6]
        rr = [_e(labD.rollout([a] * h, use_surrogate=False, init=s0d)) for a in acts6]
        if int(np.argmax(vv)) != int(np.argmax(rr)):
            T["T6_decision_H"] = h
            break
    # T7 discovery: rank 20 random forces by surrogate enstrophy vs oracle
    cand = [{"d_nu": 0.0, "force": {"amp": float(rng.uniform(-2, 2)),
                                    "angle": float(rng.uniform(0, 6.28)),
                                    "x": float(rng.random()), "y": float(rng.random()),
                                    "sigma": 0.1}} for _ in range(20 if quick else 50)]
    def ens_of(roll):
        u, v = roll[-1]["u"], roll[-1]["v"]
        return float(np.mean(u ** 2 + v ** 2))
    lab2 = Lab(BurgersOracle(), surrogate=B)
    s0 = lab2.reset(seed=seed)
    virt = [ens_of(lab2.rollout([a] * 5, use_surrogate=True, init=s0)) for a in cand]
    real = [ens_of(lab2.rollout([a] * 5, use_surrogate=False, init=s0)) for a in cand]
    k = 5
    tv = set(np.argsort(virt)[-k:])
    tr = set(np.argsort(real)[-k:])
    T["T7_R_discovery_top5"] = float(len(tv & tr) / k)
    T["T7_Delta_V2R"] = float(np.mean(np.array(virt) - np.array(real)))
    # T8 counterfactual divergence: same state, different actions.
    # A blind model predicts identical futures -> zero sensitivity and no
    # correlation with true divergence. This is the action-conditioning ablation.
    acts8 = [{"d_nu": 0.0, "force": {"amp": a, "angle": g, "x": 0.5, "y": 0.5,
                                     "sigma": 0.1}}
             for a, g in [(1.5, 0.0), (-1.5, 0.0), (1.5, np.pi / 2),
                          (-1.5, np.pi / 2), (0.2, 0.0), (2.0, np.pi),
                          (1.0, np.pi / 4), (-1.0, 3 * np.pi / 4)]]
    ref = BurgersOracle(seed=seed + 500)
    s0_8, steps8 = ref.reset(seed=seed, nu=0.01), 8
    true_ends = []
    for a in acts8:
        ref.set_state(copy_state(s0_8))
        true_ends.append(ref.rollout([a] * steps8)[-1])
    Dt = _pairwise_dists(true_ends)
    for name, model in [("A", A), ("V", V), ("B", B)]:
        if model is None:
            T[f"T8_corr_{name}"], T[f"T8_sens_{name}"] = None, None
            continue
        ends = []
        for a in acts8:
            sm = copy_state(s0_8)
            for _ in range(steps8):
                sm = model.predict(sm, a)
            ends.append(sm)
        Dm = _pairwise_dists(ends)
        T[f"T8_corr_{name}"] = (float(np.corrcoef(Dt, Dm)[0, 1])
                                if Dm.std() > 0 else 0.0)
        T[f"T8_sens_{name}"] = float(Dm.mean() / (Dt.mean() + 1e-12))
    T["V_from_weights"] = bool(v_ok)
    T["B_from_weights"] = bool(from_weights)
    return T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--retrain", action="store_true",
                   help="ignore weights_B.pt and train a small net instead "
                        "(numbers will NOT match RESULTS.json)")
    a = p.parse_args()
    print(json.dumps(run_all(quick=a.quick, retrain=a.retrain), indent=2))


if __name__ == "__main__":
    main()
