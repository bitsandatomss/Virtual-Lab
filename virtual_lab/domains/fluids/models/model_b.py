"""Model B: tiny Conv UNet one-step surrogate (torch, CPU, ~100k params)."""
import numpy as np
import torch
import torch.nn as nn

from ..oracle import rasterize_force

IN_CH = 6  # u, v, fx, fy, nu_map, obstacle


def state_action_to_tensor(state, action, n=64):
    fx, fy = rasterize_force(None if action is None else action.get("force"), n)
    nu = float(state["C"]["nu"])
    ob = state["C"].get("obstacle")
    obm = np.zeros((n, n)) if ob is None else np.asarray(ob).astype(float)
    x = np.stack([state["u"], state["v"], fx, fy,
                  np.full((n, n), nu * 20.0), obm], 0).astype(np.float32)
    return torch.from_numpy(x).unsqueeze(0)


class TinyUNet(nn.Module):
    def __init__(self, base=32, in_ch=IN_CH):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(),
                                nn.Conv2d(base, base, 3, padding=1), nn.ReLU())
        self.d1 = nn.Conv2d(base, base, 3, stride=2, padding=1)  # 64->32
        self.e2 = nn.Sequential(nn.Conv2d(base, base * 2, 3, padding=1), nn.ReLU(),
                                nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.ReLU())
        self.up = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)  # 32->64
        self.out = nn.Conv2d(base * 2, 2, 3, padding=1)

    def forward(self, x):
        f1 = self.e1(x)
        f2 = self.e2(self.d1(f1))
        u = self.up(f2)
        return self.out(torch.cat([u, f1], 1))  # residual du, dv


class SurrogateB:
    name = "B-unet"

    def __init__(self, base=32, lr=1e-3, seed=0):
        torch.manual_seed(seed)
        self.net = TinyUNet(base)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        n_params = sum(p.numel() for p in self.net.parameters())
        self.n_params = n_params

    def predict(self, state, action=None):
        self.net.eval()
        with torch.no_grad():
            x = state_action_to_tensor(state, action, state["u"].shape[0])
            d = self.net(x).squeeze(0).numpy()
        nu = float(state["C"]["nu"]) + (action.get("d_nu", 0.0) if action else 0.0)
        ob = state["C"].get("obstacle") if not (action and action.get("obstacle") is not None) \
            else action["obstacle"]
        return {"u": (state["u"] + d[0]).astype(np.float64),
                "v": (state["v"] + d[1]).astype(np.float64),
                "t": float(state["t"]) + 0.01,
                "C": {"nu": float(np.clip(nu, 0.001, 0.05)), "obstacle": ob}}

    def fit(self, data, epochs=3, batch=32):
        """data: dict/np z with Y_t (T,2,n,n), A (T,6: d_nu,amp,angle,x,y,sigma),
        Y_next, C (T: nu). Minimal MSE training."""
        self.net.train()
        Yt, Yn = data["Y_t"], data["Y_next"]
        A, C = data["A_t"], data["C"]
        n = Yt.shape[-1]
        idx = np.arange(len(Yt))
        for _ in range(epochs):
            np.random.shuffle(idx)
            for i in range(0, len(idx), batch):
                bi = idx[i:i + batch]
                xs = []
                for k in bi:
                    d_nu, amp, ang, fx_, fy_, sig = A[k]
                    force = {"amp": float(amp), "angle": float(ang),
                             "x": float(fx_), "y": float(fy_), "sigma": float(sig)}
                    fxr, fyr = rasterize_force(force, n)
                    xs.append(np.stack([Yt[k, 0], Yt[k, 1], fxr, fyr,
                                        np.full((n, n), C[k] * 20.0),
                                        np.zeros((n, n))], 0))
                x = torch.from_numpy(np.stack(xs).astype(np.float32))
                tgt = torch.from_numpy((Yn[bi] - Yt[bi]).astype(np.float32))
                self.opt.zero_grad()
                loss = self.loss_fn(self.net(x), tgt)
                loss.backward()
                self.opt.step()
        return float(loss.item())
