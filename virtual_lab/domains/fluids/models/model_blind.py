"""Model V (blind/visual predictor): same capacity as SurrogateB, NO action input.

The ablation the thesis needs: input channels are [u, v, nu_map] only —
no force fields, no d_nu. predict() accepts an action argument and ignores
it (nu passes through from state). If the environment thesis holds, V should
match B on one-step in-distribution prediction (next state is dominated by
current state) and collapse on counterfactual divergence (same state,
different actions -> V predicts identical futures).
"""
import numpy as np
import torch

from .model_b import TinyUNet

V_IN_CH = 3  # u, v, nu_map


class BlindPredictor:
    name = "V-blind"

    def __init__(self, base=32, lr=1e-3, seed=0):
        torch.manual_seed(seed)
        self.net = TinyUNet(base, in_ch=V_IN_CH)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()
        self.n_params = sum(p.numel() for p in self.net.parameters())

    def predict(self, state, action=None):
        self.net.eval()
        with torch.no_grad():
            n = state["u"].shape[0]
            nu = float(state["C"]["nu"])
            x = np.stack([state["u"], state["v"],
                          np.full((n, n), nu * 20.0)], 0).astype(np.float32)
            d = self.net(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()
        return {"u": (state["u"] + d[0]).astype(np.float64),
                "v": (state["v"] + d[1]).astype(np.float64),
                "t": float(state["t"]) + 0.01,
                "C": {"nu": nu, "obstacle": state["C"].get("obstacle")}}
