# Learned Environment Virtual Lab

> **We are not building a fluid surrogate model. We are building a virtual
> lab — a learned environment — and fluids are just the first thing in here
> to be learned about.**
>
> ```
> Reality → learned surrogate → interactive virtual world
> Reality → Surrogate → Agents → selective reality → better surrogate
> ```

A learnable domain is anything with state and controllable dynamics:
`S_{t+1} = F(S_t, A_t)`. The lab learns `Ŝ_{t+1} = G_θ(Ŝ_t, A_t)` from
interventional trajectories, then agents experiment *inside* the learned
environment — branching counterfactuals, quantifying uncertainty, and
querying reality only where the surrogate is uncertain. What matters is not
reconstructing the true mechanism but preserving the behavior that matters
under the interventions you care about (*experimental sufficiency*).

**Philosophy:** see `VISION.md` for the quarantine philosophy — surrogate vs
simulation, experimental sufficiency, honest failure modes, and what we are
and are not claiming.

## Structure

```
virtual-labs/
  virtual_lab/                 core framework (domain-agnostic)
    lab.py                     Lab API: reset / intervene / branch / rollout /
                               measure / validated_horizon / ask_surrogate
    metrics_v2r.py             Δ_V2R, R_discovery, H_eps
    active_loop.py             surrogate proposes (virtual) → oracle validates
    domains/
      fluids/                  DOMAIN #1 — 2D Burgers oracle + surrogates A/B/C,
                               intervention benchmark T1–T7, 10k→10 agent demo,
                               trained weights + RESULTS, 1B Colab notebook
      microstructure/          planned — phase-field oracle (OPMD data)
      reactions/               planned — ORD/ORDerly conditions→yield, then sequential
      pusher/                  planned — 2D robotic pushing world model
```

Every domain speaks the same core API and is judged by the same bar:
counterfactual validity (T1–T7), validated horizon `H_eps`, and
virtual-to-real transfer (`Δ_V2R`, `R_discovery`) — never prediction loss alone.

## Quickstart (fluids first)

From `virtual-labs/`:

```bash
python -c "import virtual_lab; print('ok')"
python -m virtual_lab.domains.fluids.benchmarks --quick
python -m virtual_lab.domains.fluids.agent_example --quick
```

Full story (vision → results): `virtual_lab/domains/fluids/README.md`.
Scale-up path (10M → 100M → 1B LoRA): `virtual_lab/domains/fluids/train_1B_fluids_colab.ipynb`.

## Adding a domain

1. Oracle with `reset / get_state / set_state / step / rollout` over a state
   dict (see `domains/fluids/oracle.py`), PLUS oracle-owned state ops the
   Lab delegates to (never assumes): `copy_state(s)`, `distance(a, b)`,
   `measure(s)`. Domain params (e.g. viscosity) pass through
   `Lab.reset(seed, **kw)` — the core API takes no domain kwargs itself.
2. `data_gen` sampling the **action space**, not just initial conditions —
   `P(Y|do(A))` needs interventional coverage.
3. Surrogate(s) exposing `predict(state, action) -> state`.
4. Benchmark + agent demo reporting `H_eps`, `Δ_V2R`, `R_discovery`.

The scarce resource is access to reality; the lab turns one physical world
into arbitrarily many resettable virtual copies, with reality as teacher
and validator.
