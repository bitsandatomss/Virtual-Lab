"""Model A: persistence + coarse baseline (no training)."""
import numpy as np


class PersistenceBaseline:
    name = "A-persistence"

    def predict(self, state, action=None):
        s = {"u": state["u"].copy(), "v": state["v"].copy(),
             "t": float(state["t"]) + 0.01,
             "C": {"nu": float(state["C"]["nu"]),
                   "obstacle": state["C"].get("obstacle")}}
        if action and action.get("d_nu"):
            s["C"]["nu"] = float(np.clip(s["C"]["nu"] + action["d_nu"], 0.001, 0.05))
        return s


class CoarseBaseline:
    """Persistence through 2x coarse-graining (diffusive proxy)."""
    name = "A-coarse"

    def predict(self, state, action=None):
        u, v = state["u"], state["v"]
        uc = (u[0::2, 0::2] + u[1::2, 0::2] + u[0::2, 1::2] + u[1::2, 1::2]) / 4
        vc = (v[0::2, 0::2] + v[1::2, 0::2] + v[0::2, 1::2] + v[1::2, 1::2]) / 4
        s = {"u": np.repeat(np.repeat(uc, 2, 0), 2, 1),
             "v": np.repeat(np.repeat(vc, 2, 0), 2, 1),
             "t": float(state["t"]) + 0.01,
             "C": {"nu": float(state["C"]["nu"]),
                   "obstacle": state["C"].get("obstacle")}}
        if action and action.get("d_nu"):
            s["C"]["nu"] = float(np.clip(s["C"]["nu"] + action["d_nu"], 0.001, 0.05))
        return s


BaselineA = PersistenceBaseline  # default A
