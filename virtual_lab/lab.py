"""Core Lab API: reset / intervene / branch / rollout / measure / validated_horizon / ask_surrogate.

Domain-agnostic: the Lab never assumes what a state looks like. All
state-shaped operations are owned by the domain oracle and reached through
duck-typed delegation with generic fallbacks:

- oracle.copy_state(s)  (fallback: deepcopy — correct for any state dict)
- oracle.distance(a, b) (fallback: state_error, the legacy fluids default)
- oracle.measure(s)     (no fallback: silent fluid metrics on a chemistry
  state would be worse than an explicit error)

The oracle is reality; an optional surrogate enables virtual experiments.

Cursor conventions (the lab must never corrupt a live experiment):
- Probing is side-effect-free: rollout(..., init=s) and validated_horizon
  always restore the live oracle cursor, including on early return.
- intervene() is the ONLY op that advances the live oracle state.
"""
import copy as _copy

import numpy as np


def copy_state(s):
    """Generic independent copy of a state dict (arrays, scalars, nesting)."""
    return _copy.deepcopy(s)


def state_error(a, b):
    """Legacy default metric: MSE over u/v fields.

    Kept for RESULTS comparability and as the Lab.distance fallback; new
    domains should define oracle.distance() instead of importing this.
    """
    return float(np.mean((a["u"] - b["u"]) ** 2 + (a["v"] - b["v"]) ** 2))


class Lab:
    def __init__(self, oracle, surrogate=None):
        if oracle is None:
            raise ValueError("pass a domain oracle, e.g. Lab(BurgersOracle())")
        self.oracle = oracle
        self.surrogate = surrogate  # must expose predict(state, action)->state

    # -- state ops (oracle-owned, generic fallbacks) --
    def copy(self, state):
        copier = getattr(self.oracle, "copy_state", None)
        return copier(state) if copier is not None else copy_state(state)

    def branch(self, state):
        """Return an independent copy (fork) of a state."""
        return self.copy(state)

    def distance(self, a, b):
        metric = getattr(self.oracle, "distance", None)
        return metric(a, b) if metric is not None else state_error(a, b)

    def measure(self, state):
        measurer = getattr(self.oracle, "measure", None)
        if measurer is None:
            raise AttributeError(
                "oracle has no measure(state); define domain observables "
                "on the oracle instead of leaking them into the Lab")
        return measurer(state)

    # -- core API --
    def reset(self, seed=None, **kw):
        """Reset the oracle. Domain params (e.g. nu) pass through as kwargs."""
        return self.oracle.reset(seed=seed, **kw)

    def intervene(self, action):
        """Single do(A) step on the live oracle. Returns next state.

        This is the ONLY op that advances the live experiment cursor.
        """
        return self.oracle.step(action)

    def rollout(self, actions, steps=None, use_surrogate=False, init=None):
        """Roll actions forward.

        Surrogate path never touches the oracle. Oracle path with init
        probes from a fork and restores the live cursor afterwards.
        Oracle path without init advances the live experiment (like
        repeated intervene calls).
        """
        if isinstance(actions, dict) or actions is None:
            actions = [actions] * (steps or 1)
        if not use_surrogate or self.surrogate is None:
            if init is not None:
                prev = self.oracle.get_state()
                try:
                    self.oracle.set_state(self.copy(init))
                    return self.oracle.rollout(actions)
                finally:
                    self.oracle.set_state(prev)
            return self.oracle.rollout(actions)
        # surrogate rollout from init (or current oracle state, read-only)
        s = self.copy(init) if init is not None else self.oracle.get_state()
        out = []
        for a in actions:
            s = self.surrogate.predict(s, a)
            out.append(self.copy(s))
        return out

    def ask_surrogate(self, state, action):
        if self.surrogate is None:
            raise RuntimeError("no surrogate attached")
        return self.surrogate.predict(state, action)

    # -- analysis --
    def validated_horizon(self, actions, eps=1e-3, init=None):
        """Steps until surrogate||oracle error exceeds eps. inf if no surrogate.

        Side-effect-free: the live oracle cursor is restored on every path,
        including early return.
        """
        if self.surrogate is None:
            return float("inf")
        if isinstance(actions, dict) or actions is None:
            actions = [actions] * 50
        prev = self.oracle.get_state()
        try:
            s_or = self.copy(init) if init is not None else self.copy(prev)
            s_su = self.copy(s_or)
            for h, a in enumerate(actions):
                self.oracle.set_state(self.copy(s_or))
                s_or = self.oracle.step(a)
                s_su = self.surrogate.predict(s_su, a)
                if self.distance(s_or, s_su) > eps:
                    return h + 1
            return len(actions)
        finally:
            self.oracle.set_state(prev)
