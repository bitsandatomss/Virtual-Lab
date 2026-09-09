"""Smoke tests for the Learned Environment Virtual Lab.

Fast, deterministic, CPU-only. Run from virtual-labs/:
    python -m pytest tests/ -q
"""
import numpy as np
import torch

from virtual_lab import Lab, ActiveLoop, delta_v2r, r_discovery, h_eps
from virtual_lab.domains.fluids.oracle import BurgersOracle
from virtual_lab.domains.fluids.data_gen import generate
from virtual_lab.domains.fluids.models.model_a import PersistenceBaseline
from virtual_lab.domains.fluids.models.model_b import SurrogateB


def test_oracle_deterministic():
    o1, o2 = BurgersOracle(seed=7), BurgersOracle(seed=7)
    o1.reset(nu=0.01)
    o2.reset(nu=0.01)
    a = {"d_nu": 0.0, "force": {"amp": 1.0, "angle": 0.5, "x": 0.5,
                                "y": 0.5, "sigma": 0.1}}
    for _ in range(5):
        s1, s2 = o1.step(a), o2.step(a)
    assert np.array_equal(s1["u"], s2["u"])
    assert s1["u"].shape == (64, 64)


def test_oracle_nu_clip():
    o = BurgersOracle()
    o.reset(nu=0.01)
    o.step({"d_nu": 10.0, "force": None})
    assert o.nu <= 0.05
    o.step({"d_nu": -10.0, "force": None})
    assert o.nu >= 0.001


def test_lab_requires_oracle():
    try:
        Lab(None)
    except ValueError:
        return
    raise AssertionError("Lab(None) must raise ValueError")


def test_lab_branch_independent():
    lab = Lab(BurgersOracle(seed=0))
    s0 = lab.reset(seed=0)
    b = lab.branch(s0)
    b["u"] += 999.0
    assert not np.array_equal(s0["u"], b["u"])


def test_active_loop_requires_lab():
    try:
        ActiveLoop(None)
    except ValueError:
        return
    raise AssertionError("ActiveLoop(None) must raise ValueError")


def test_surrogate_predict_contract():
    lab = Lab(BurgersOracle(seed=1))
    s = lab.reset(seed=1, nu=0.01)
    B = SurrogateB(seed=0)
    p = B.predict(s, {"d_nu": 0.0, "force": None})
    assert p["u"].shape == (64, 64) and p["v"].shape == (64, 64)
    assert abs(p["t"] - (s["t"] + 0.01)) < 1e-12
    assert 0.001 <= p["C"]["nu"] <= 0.05
    base = PersistenceBaseline()
    assert base.predict(s, None)["u"].shape == (64, 64)


def test_metrics():
    assert delta_v2r([1.0, 2.0], [1.0, 1.0]) == 0.5
    assert r_discovery({1, 2}, {2, 3, 4}) == 1 / 3
    assert h_eps([1e-4, 2e-3, 1e-2], 1e-3) == 1
    assert h_eps([1e-4], 1e-3) == 1


def test_weights_load():
    from virtual_lab.domains.fluids.benchmarks import WEIGHTS_B
    assert WEIGHTS_B.exists(), "weights_B.pt must ship with the repo"
    B = SurrogateB(seed=0)
    B.net.load_state_dict(torch.load(str(WEIGHTS_B), map_location="cpu",
                                     weights_only=True))
    lab = Lab(BurgersOracle(seed=2))
    s = lab.reset(seed=2)
    p = B.predict(s, {"d_nu": 0.0, "force": None})
    assert np.isfinite(p["u"]).all()


def test_data_gen_shapes():
    d = generate(trajs=2, steps=3, seed=0)
    assert d["Y_t"].shape == (6, 2, 64, 64)
    assert d["A_t"].shape == (6, 6)
    assert d["Y_next"].shape == (6, 2, 64, 64)


def test_blind_ignores_action():
    from virtual_lab.domains.fluids.models.model_blind import BlindPredictor
    lab = Lab(BurgersOracle(seed=3))
    s = lab.reset(seed=3)
    V = BlindPredictor(seed=0)
    a1 = {"d_nu": 0.0, "force": {"amp": 2.0, "angle": 0.0, "x": 0.5,
                                 "y": 0.5, "sigma": 0.1}}
    a2 = {"d_nu": 0.0, "force": {"amp": -2.0, "angle": 2.1, "x": 0.1,
                                 "y": 0.9, "sigma": 0.05}}
    p1, p2 = V.predict(s, a1), V.predict(s, a2)
    assert np.array_equal(p1["u"], p2["u"])
    B = SurrogateB(seed=0)
    q1, q2 = B.predict(s, a1), B.predict(s, a2)
    assert not np.array_equal(q1["u"], q2["u"])  # sighted model must respond


def test_validated_horizon_restores_cursor():
    lab = Lab(BurgersOracle(seed=5), surrogate=PersistenceBaseline())
    s_live = lab.reset(seed=5, nu=0.01)
    a = {"d_nu": 0.0, "force": {"amp": 1.0, "angle": 0.5, "x": 0.5,
                                "y": 0.5, "sigma": 0.1}}
    lab.intervene(a)
    lab.intervene(a)
    snap_u = lab.oracle.get_state()["u"].copy()
    s0 = BurgersOracle(seed=99).reset(seed=99, nu=0.01)
    lab.validated_horizon([a] * 6, eps=1e-3, init=s0)
    assert np.array_equal(lab.oracle.get_state()["u"], snap_u)


def test_rollout_with_init_does_not_move_cursor():
    lab = Lab(BurgersOracle(seed=6))
    lab.reset(seed=6, nu=0.01)
    a = {"d_nu": 0.0, "force": {"amp": 1.0, "angle": 0.5, "x": 0.5,
                                "y": 0.5, "sigma": 0.1}}
    snap_u = lab.oracle.get_state()["u"].copy()
    s0 = BurgersOracle(seed=77).reset(seed=77, nu=0.01)
    out = lab.rollout([a] * 4, use_surrogate=False, init=s0)
    assert len(out) == 4
    assert np.array_equal(lab.oracle.get_state()["u"], snap_u)


def test_reset_forwards_domain_kwargs():
    lab = Lab(BurgersOracle(seed=8))
    s = lab.reset(seed=8, nu=0.02)
    assert abs(s["C"]["nu"] - 0.02) < 1e-12


def test_measure_delegates_to_oracle():
    lab = Lab(BurgersOracle(seed=9))
    s = lab.reset(seed=9, nu=0.01)
    m = lab.measure(s)
    assert set(m) == {"ke", "enstrophy", "mean_u", "mean_v", "t"}
    assert m["ke"] >= 0.0 and m["enstrophy"] >= 0.0


def test_lab_distance_matches_oracle():
    from virtual_lab import state_error
    lab = Lab(BurgersOracle(seed=10))
    s1 = lab.reset(seed=10, nu=0.01)
    s2 = lab.reset(seed=11, nu=0.01)
    assert lab.distance(s1, s2) == state_error(s1, s2)
