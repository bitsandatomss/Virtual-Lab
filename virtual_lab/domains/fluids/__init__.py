"""Fluids domain — one learnable thing inside the virtual lab.

Oracle: 2D viscous Burgers (numpy, 64x64 periodic). Surrogates A/B/C learn
``s_{t+1} = G(s_t, a_t)`` from interventional trajectories.
"""
from .oracle import BurgersOracle, rasterize_force
from .data_gen import generate
from .models.model_a import PersistenceBaseline, CoarseBaseline
from .models.model_b import SurrogateB
from .models.model_c import EnsembleC
from .models.model_blind import BlindPredictor

__all__ = ["BurgersOracle", "rasterize_force", "generate",
           "PersistenceBaseline", "CoarseBaseline", "SurrogateB", "EnsembleC",
           "BlindPredictor"]
