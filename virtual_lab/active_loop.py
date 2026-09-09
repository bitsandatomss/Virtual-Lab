"""Core active loop: surrogate-guided proposal + oracle validation."""
import numpy as np


def random_candidates(rng, k, sigma=0.1):
    return [{"d_nu": 0.0, "force": {"amp": float(rng.uniform(-2, 2)),
                                    "angle": float(rng.uniform(0, 2 * np.pi)),
                                    "x": float(rng.random()), "y": float(rng.random()),
                                    "sigma": sigma}} for _ in range(k)]


class ActiveLoop:
    def __init__(self, lab, score_fn=None):
        if lab is None:
            raise ValueError("pass a Lab bound to a domain oracle")
        self.lab = lab
        self.score_fn = score_fn or (lambda states: self.lab.measure(states[-1])["enstrophy"])

    def propose(self, state, cands, unc_fn=None):
        """Score candidates virtually; optionally add uncertainty bonus."""
        scored = []
        for a in cands:
            traj = self.lab.rollout([a] * 5, use_surrogate=True, init=state)
            s = self.score_fn(traj)
            if unc_fn is not None:
                s = s + float(unc_fn(state, a))
            scored.append((s, a))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored

    def validate(self, state, actions, steps=5):
        """Ground top actions on the oracle."""
        out = []
        for a in actions:
            traj = self.lab.rollout([a] * steps, use_surrogate=False, init=state)
            out.append((self.score_fn(traj), a))
        out.sort(key=lambda t: t[0], reverse=True)
        return out

    def run(self, iters=3, per_iter=200, topk=10, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        s0 = self.lab.reset(seed=seed)
        log = []
        for it in range(iters):
            cands = random_candidates(rng, per_iter)
            unc = (lambda s, a: self.lab.surrogate.uncertainty(s, a)["var"]
                   if hasattr(self.lab.surrogate, "uncertainty") else None)
            ranked = self.propose(s0, cands, unc_fn=unc)
            top = [a for _, a in ranked[:topk]]
            real = self.validate(s0, top)
            log.append({"iter": it, "virt_best": ranked[0][0], "real_best": real[0][0]})
            if verbose:
                print(f"iter {it}: virt={ranked[0][0]:.4f} real={real[0][0]:.4f}")
        return log
