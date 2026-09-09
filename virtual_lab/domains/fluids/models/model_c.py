"""Model C: ensemble wrapper around B with variance + OOD flag."""
import numpy as np
from .model_b import SurrogateB


class EnsembleC:
    name = "C-ensemble"

    def __init__(self, k=3, base=32, ood_thresh=1e-3, **kw):
        self.members = [SurrogateB(base=base, seed=s, **kw) for s in range(k)]
        self.ood_thresh = float(ood_thresh)

    def predict(self, state, action=None):
        preds = [m.predict(state, action) for m in self.members]
        us = np.stack([p["u"] for p in preds])
        vs = np.stack([p["v"] for p in preds])
        mean = {"u": us.mean(0), "v": vs.mean(0), "t": preds[0]["t"], "C": preds[0]["C"]}
        var = float((us.var(0) + vs.var(0)).mean())
        return mean, {"var": var, "ood": bool(var > self.ood_thresh)}

    # Lab-compatible: predict returns mean state
    def predict_mean(self, state, action=None):
        return self.predict(state, action)[0]

    def uncertainty(self, state, action=None):
        return self.predict(state, action)[1]

    def fit(self, data, **kw):
        return [m.fit(data, **kw) for m in self.members]
