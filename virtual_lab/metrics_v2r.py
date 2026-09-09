"""Metrics: Delta_V2R, R_discovery, H_eps."""
import numpy as np


def delta_v2r(virtual_scores, real_scores):
    """Virtual-to-real gap: mean(virtual - real). Positive = overoptimism."""
    v = np.asarray(virtual_scores, float)
    r = np.asarray(real_scores, float)
    return float((v - r).mean())


def r_discovery(found, total):
    """Fraction of true top-k recovered. found/total: ints or sets."""
    if isinstance(found, (set, list)) and isinstance(total, (set, list)):
        return float(len(set(found) & set(total)) / max(1, len(total)))
    return float(found / max(1, total))


def h_eps(errors, eps):
    """Validated horizon: first index where error > eps (len if never)."""
    for h, e in enumerate(errors):
        if e > eps:
            return h
    return len(errors)
