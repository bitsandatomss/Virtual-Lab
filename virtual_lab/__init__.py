"""Learned Environment Virtual Lab — core framework.

Domain-agnostic pieces: the Lab API (reset/intervene/branch/rollout/measure),
virtual-to-real metrics, and the active propose/validate loop.
Learnable domains live in ``virtual_lab.domains.*`` — fluids is the first.
"""
from .lab import Lab, copy_state, state_error
from .metrics_v2r import delta_v2r, r_discovery, h_eps
from .active_loop import ActiveLoop, random_candidates

__all__ = ["Lab", "copy_state", "state_error", "delta_v2r", "r_discovery",
           "h_eps", "ActiveLoop", "random_candidates"]
__version__ = "0.2.0"
