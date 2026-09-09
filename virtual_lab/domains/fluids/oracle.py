"""Oracle: 2D viscous Burgers, numpy CPU, 64x64 periodic.

State s_t = {u, v, t, C}; C = {nu, obstacle mask|None}.
Action a_t = {d_nu, force{amp,angle,x,y,sigma}|None, obstacle|None}.
"""
import numpy as np

N = 64
NU_RANGE = (0.001, 0.05)


def rasterize_force(force, n=N):
    """Gaussian force blob -> (fx, fy) fields. force None -> zeros."""
    fx = np.zeros((n, n), dtype=np.float64)
    fy = np.zeros((n, n), dtype=np.float64)
    if force is None:
        return fx, fy
    amp = float(force.get("amp", 0.0))
    ang = float(force.get("angle", 0.0))
    cx = float(force.get("x", 0.5)) * n
    cy = float(force.get("y", 0.5)) * n
    sig = float(force.get("sigma", 0.08)) * n + 1e-6
    yy, xx = np.mgrid[0:n, 0:n]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))
    fx = amp * np.cos(ang) * g
    fy = amp * np.sin(ang) * g
    return fx, fy


def _grad_x(f, dx):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)


def _grad_y(f, dx):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)


def _laplacian(f, dx):
    return (np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0)
            + np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1)
            - 4 * f) / (dx ** 2)


class BurgersOracle:
    def __init__(self, n=N, nu=0.01, dt=0.01, seed=0):
        self.n = n
        self.nu = float(nu)
        self.dt = float(dt)
        self.dx = 2 * np.pi / n
        self.rng = np.random.default_rng(seed)
        self.u = np.zeros((n, n))
        self.v = np.zeros((n, n))
        self.t = 0.0
        self.obstacle = None  # bool mask or None

    def reset(self, seed=None, nu=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if nu is not None:
            self.nu = float(np.clip(nu, *NU_RANGE))
        # random smooth IC: few Fourier modes
        kx = self.rng.integers(1, 4, size=3)
        ky = self.rng.integers(1, 4, size=3)
        ph = self.rng.uniform(0, 2 * np.pi, size=3)
        yy, xx = np.mgrid[0:self.n, 0:self.n] * self.dx
        self.u = np.zeros_like(xx, dtype=np.float64)
        self.v = np.zeros_like(xx, dtype=np.float64)
        for k in range(3):
            self.u += np.sin(kx[k] * xx + ph[k]) * np.cos(ky[k] * yy)
            self.v += np.cos(kx[k] * xx) * np.sin(ky[k] * yy + ph[k])
        self.u *= 0.5
        self.v *= 0.5
        self.t = 0.0
        return self.get_state()

    def get_state(self):
        return {"u": self.u.copy(), "v": self.v.copy(), "t": float(self.t),
                "C": {"nu": self.nu,
                      "obstacle": None if self.obstacle is None else self.obstacle.copy()}}

    def set_state(self, s):
        self.u = s["u"].copy()
        self.v = s["v"].copy()
        self.t = float(s["t"])
        self.nu = float(s["C"]["nu"])
        ob = s["C"].get("obstacle")
        self.obstacle = None if ob is None else ob.copy()

    # -- Lab-facing state ops (domain-owned; the Lab delegates here) --
    def copy_state(self, s):
        return {"u": s["u"].copy(), "v": s["v"].copy(), "t": float(s["t"]),
                "C": {"nu": float(s["C"]["nu"]),
                      "obstacle": None if s["C"].get("obstacle") is None
                      else s["C"]["obstacle"].copy()}}

    def distance(self, a, b):
        return float(np.mean((a["u"] - b["u"]) ** 2 + (a["v"] - b["v"]) ** 2))

    def measure(self, state):
        u, v = state["u"], state["v"]
        ke = float(0.5 * np.mean(u ** 2 + v ** 2))
        # enstrophy proxy via vorticity energy (periodic grid)
        n = u.shape[0]
        dx = 2 * np.pi / n
        dvdx = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2 * dx)
        dudy = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2 * dx)
        ens = float(0.5 * np.mean((dvdx - dudy) ** 2))
        return {"ke": ke, "enstrophy": ens, "mean_u": float(u.mean()),
                "mean_v": float(v.mean()), "t": float(state["t"])}

    def apply_action(self, action):
        """Apply do(A): update nu / obstacle. Returns (fx, fy)."""
        if action is None:
            return np.zeros_like(self.u), np.zeros_like(self.u)
        if "d_nu" in action and action["d_nu"]:
            self.nu = float(np.clip(self.nu + action["d_nu"], *NU_RANGE))
        if "obstacle" in action and action["obstacle"] is not None:
            self.obstacle = np.asarray(action["obstacle"]).astype(bool)
        return rasterize_force(action.get("force"), self.n)

    def step(self, action=None):
        fx, fy = self.apply_action(action)
        dudx, dudy = _grad_x(self.u, self.dx), _grad_y(self.u, self.dx)
        dvdx, dvdy = _grad_x(self.v, self.dx), _grad_y(self.v, self.dx)
        rhs_u = -self.u * dudx - self.v * dudy + self.nu * _laplacian(self.u, self.dx) + fx
        rhs_v = -self.u * dvdx - self.v * dvdy + self.nu * _laplacian(self.v, self.dx) + fy
        self.u = self.u + self.dt * rhs_u
        self.v = self.v + self.dt * rhs_v
        if self.obstacle is not None:
            self.u[self.obstacle] = 0.0
            self.v[self.obstacle] = 0.0
        # NS flag optional: clip blowups
        np.clip(self.u, -10, 10, out=self.u)
        np.clip(self.v, -10, 10, out=self.v)
        self.t += self.dt
        return self.get_state()

    def rollout(self, actions, steps=None):
        """Roll oracle forward. actions: list or single dict or None."""
        if isinstance(actions, dict) or actions is None:
            actions = [actions] * (steps or 1)
        out = []
        for a in actions:
            out.append(self.step(a))
        return out
