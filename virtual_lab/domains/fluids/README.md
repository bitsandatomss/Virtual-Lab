# Fluids domain — first learnable thing in the virtual lab

> Part of the **Learned Environment Virtual Lab** — see `../../README.md`.
> This project is not a fluid surrogate model. It is a virtual lab;
> fluids just happen to be the first thing in here to be learned about.

> **Thesis: learn a sufficiently faithful surrogate of an environment, then interact with that surrogate as though it were the environment.**
>
> ```
> Reality → learned surrogate → interactive virtual world
> Reality → Surrogate → Agents → selective reality → better surrogate
> ```

This repo is a working minimal instance of that thesis: a 2D fluid environment
learned from interventional trajectories, with branching, uncertainty, an
intervention benchmark, virtual-to-real validation, and an agent demo that runs
10,000 virtual experiments and validates the top 10 against the oracle.

## 1. Surrogate vs simulation — the core distinction

A conventional simulator starts from an explicit mechanism:

```
rules → state transition → next state        s_{t+1} = F_θ(s_t, a_t)
```

A learned surrogate learns the transition from observations/trajectories:

```
observations → learned structure → predicted next state    ŝ_{t+1} = G_φ(o_{≤t}, a_t)
```

A surrogate *is* a simulator in the functional sense. The distinction that
matters is **explicitly specified mechanism vs empirically learned behavioral
approximation**. The surrogate wins when it preserves the
intervention-relevant behavior at a fraction of the query cost — this is
*experimental sufficiency*: you don't need the whole microscopic state
`X ∈ R^N`, only the map from interventions you care about to outcomes you
measure, possibly on a much lower-dimensional manifold (`Z ∈ R^d, d ≪ N`).

Two economics make this powerful:

- **Amortization.** Training costs `N·C_s` once; each virtual experiment costs
  `C_g ≪ C_s`. Total `C_train + M·C_g` vs `M·C_s` — the surrogate wins when an
  AI scientist runs `M = 10^6+` counterfactual queries.
- **Virtualization.** One physical world becomes arbitrarily many resettable
  virtual copies. Reality turns from the place you experiment into the
  teacher/validator of the place you experiment.

## 2. Why fluids first (not chemistry)

Real chemistry on day one would drown the project in instrumentation cost.
2D fluid dynamics is the ideal first oracle: explicit ground truth,
controllable parameters, continuous state, rich chaotic dynamics, expensive
solvers, cheap data generation, and obvious visualization. It is also the
bridge to the Demis/Veo intuition: Navier–Stokes is traditionally
theory → equations → solver, while a video model can produce plausible fluid
behavior from observations alone. This repo tests whether that plausibility
can be pushed to **validated, intervention-capable** prediction —
`P(Y|do(A))`, not just `P(Y|A)` or pretty frames.

## 3. Architecture

```
                         REALITY (oracle)
                              │  experiments / trajectories τ = (s_0,a_0,s_1,…)
                              ▼
                 ┌─────────────────────────┐
                 │   LEARNED SURROGATE      │
                 │   dynamics + state +     │
                 │   uncertainty + actions  │
                 └────────────┬────────────┘
                              │  virtual experiments (branch 10⁴–10⁶)
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
         hypothesis A   hypothesis B    design search
               └──────────────┼──────────────┘
                              ▼  experiment selection (uncertainty / info-per-dollar)
                           REAL LAB (oracle validation)
                              └──────► update surrogate (active learning)
```

**Oracle** (`oracle.py`): 2D viscous Burgers, numpy, 64×64 periodic,
Euler + central differences. State `s_t = {u, v, t, C}`,
`C = {nu ∈ [0.001, 0.05], obstacle mask}`. Action
`a_t = {d_nu, force{amp, angle, x, y, sigma}, obstacle}`. Ground truth =
"reality" for virtual-to-real transfer.

**Dataset** (`data_gen.py`): trajectories built around *interventions* —
random `do(A)` force/viscosity sequences, not just random initial conditions.
Full set: 200 trajs × 50 steps = 10,000 `(Y_t, A_t, Y_next, C)` pairs, plus
holdout (seed 123) and OOD obstacle / high-viscosity splits. Passive
observation alone cannot identify `P(Y|do(A))`; coverage of the action space
is the point ordinary video pretraining misses.

**Three models, not one:**

| Model | File | What it is | Role |
|---|---|---|---|
| A — baseline | `models/model_a.py` | persistence + coarse-grain, no training | mechanistic/persistence control |
| B — state surrogate | `models/model_b.py` | tiny Conv UNet (~110k params), `(Y_t,a_t,C)→Y_{t+1}` residual | interpolative surrogate (Level 2) |
| C — latent/uncertain env | `models/model_c.py` | 3× ensemble of B + predictive variance + OOD flag | counterfactual + scientific surrogate (Levels 3–5) |

The A/B/C ablation tests the manifold hypothesis directly:
explicit state → learned dynamics → uncertainty-aware environment
(the planned progression continues to latent-state and pixels-only).

**Lab API** (`lab.py`): `reset / intervene (do(A)) / branch (fork state) /
rollout / measure (ke, enstrophy, sensors) / validated_horizon / ask_surrogate`.
Branching is what makes it a laboratory instead of a predictor:

```
              current fluid
                   │
        ┌──────────┼──────────┐
        ↓          ↓          ↓
     do(A₁)     do(A₂)     do(A₃)
        │          │          │
        ↓          ↓          ↓
     world 1   world 2   world 3  → compare
```

**Agent + active loop** (`agent_example.py`, `active_loop.py`):
hypothesize → branch thousands of virtual experiments → pick top-k →
validate on oracle → update. The loop is
`observe → learn → simulate → uncertainty → acquire → oracle → update`,
so the surrogate is an epistemic instrument that decides which reality to
query next, not a frozen predictor.

## 4. Benchmark: interventions, not pixels

MSE/PSNR/FID can look excellent while the environment is scientifically
useless. The suite (`benchmarks.py`, `metrics_v2r.py`) tests counterfactual
validity instead:

- **T1** one-step accuracy (held-out ICs)
- **T2** rollout stability (20 steps; catches compounding)
- **T3** interventional holdout (unseen action combinations)
- **T4** obstacle OOD + ensemble-variance flag (must detect, not silently fail)
- **T5** viscosity extrapolation (nu = 0.05)
- **T6** validated horizon `H_eps` = steps until surrogate‖oracle error > eps —
  the trust radius; the lab must know where it stops trusting itself
- **T7** discovery: rank candidates virtually, check overlap with oracle ranking

The two project-defining metrics:

- `Δ_V2R = mean(virtual − real)` on selected candidates (over-optimism if
  positive). Small |Δ| + high overlap = ranks like reality.
- `R_discovery` = fraction of oracle top-k recovered by virtual screening —
  the decision-relevant quantity: *how much scientific search did the virtual
  environment replace?*

## 5. Results (fully trained)

Config: 64×64, seed 0, B 25 epochs / C 15 epochs, batch 64 (`RESULTS.json`,
`RESULTS.md`, `train_log.json`).

| test | A (baseline) | V (blind) | B (sighted) | C (ensemble) |
|---|---|---|---|---|
| T1 one-step MSE | — | 1.15e-05 | 2.15e-06 | 1.72e-06 |
| T2 rollout-20 MSE | 1.24e-02 | — | 9.71e-04 | 8.95e-04 |
| T3 in-dist holdout | — | — | 4.92e-05 | 6.77e-05 |
| T3 shift (amp=3.0) | 6.23e-05 | 2.50e-05 | 4.49e-06 | — |
| T4 obstacle OOD | — | — | 7.46e-05 | 7.40e-05 |
| T5 nu=0.05 rollout | — | — | 5.07e-04 | 3.78e-04 |
| T6 H(eps=1e-3) | — | — | 13 | 13 |
| T6 decision H | 1 | 1 | 1 | — |
| T7 R_discovery (top-5) | — | — | 0.40 | 0.80 |
| T7 Δ_V2R | — | — | +1.07e-03 | −9.30e-04 |
| **T8 counterfactual corr** | **0.0** | **0.0** | **0.997** | — |
| **T8 sensitivity** | **0.0** | **0.0** | **1.07** | — |

**T8 is the action-conditioning ablation:** same initial state, 8 different
force actions. The blind model (V, no action input) predicts *identical*
futures — corr=0.0, sens=0.0. The sighted model (B) tracks truth at
corr=0.997, sens=1.07. This proves action conditioning is what makes it
an environment, not a predictor. (V actually fits one-step *better* than B
but is useless for counterfactuals — prediction accuracy ≠ environment
fidelity.)

**Calibration:** reliability_corr=0.986 (variance tracks error well);
OOD AUROC=0.500 (3 near-identical seeds → insufficient diversity for OOD
detection via ensemble variance alone).

**Hard regime (nu=0.001):** 26% error increase vs nu=0.01 — mild because
64×64 can't resolve sharp shocks.

**Amortization:** CPU oracle 0.73ms/step, surrogate 12.3ms/step. The
oracle is *faster* — amortization is infinite here. The surrogate's value
is enabling 10k virtual screens (36.6s equivalent), not speed. For expensive
oracles (3D CFD, wet-lab), the surrogate becomes the only option.

**Agent demo (10,000 virtual → top-10 validated):** virtual-best 0.5013,
real-best 0.4987, **Δ_V2R = +2.98e-03**, **R_discovery = 0.60** — a 1000×
reduction in oracle queries per discovery round with rank-preserving transfer.
One-step 99.9% accuracy alone would not imply this; T2/T6/T7/T8 do.

## 6. The bigger picture: Learned Environments / Surrogate Worlds

Science is one application of the primitive
`Ŝ_{t+1} = G_θ(Ŝ_t, A_t)`. The same substrate virtualizes anything with
state + controllable dynamics: robots practicing millions of failures,
aircraft/fab recipes searched virtually before manufacturing, corporate
sandboxes (price/hire/launch as interventions), traffic grids and cities run
through 10⁵ futures, strategic games with adaptive opponents, students doing
experiments inside responsive worlds, games as learned worlds rather than
scripts, personal counterfactuals, and learned software environments where AI
tests deployments without touching production. The stack inverts from
`model → agent → environment` to
`reality → [learned environment] → agent` — agents live, train, design, and
plan *inside* the learned world, with reality as validator.

Philosophically this extends the MIT Bits-and-Atoms program
(bits ↔ atoms) with a middle layer: **atoms → bits → learned worlds →
selected bits → atoms**. Unlike a CAD file or equation (a description), a
learned environment is a generative, intervenable substrate: you `do(a)` in
it and get different worlds. Unlike a traditional digital twin (this machine,
now), it supports counterfactual branching (this machine, had history gone
otherwise, ×10⁶ copies). Wheeler's "it from bit" stays metaphysics; the
engineering claim here is narrower and sufficient: *much intervention-relevant
behavior may be computationally representable from observations without the
full mechanistic theory* — prediction and explanation separate, and AI
exploits that separation at scale.

## 7. Repo layout

```
domains/fluids/
  oracle.py            Burgers oracle (reality), rasterize_force
  data_gen.py          random do(A) trajectories → .npz
  models/model_a.py    persistence + coarse baseline
  models/model_b.py    tiny UNet surrogate (~110k params)
  models/model_blind.py blind predictor (no action input — T8 ablation)
  models/model_c.py    3-ensemble + variance / OOD flag
  benchmarks.py        Tests 1–8 (T8 = counterfactual divergence ablation)
  metrics_v2r.py       delta_v2r, r_discovery, h_eps
  active_loop.py       propose (virtual) + validate (oracle) loop
  agent_example.py     CEM planner + 10k virtual screen demo
  planner.py           CEM in-surrogate vs random shooting
  active_demo.py       uncertainty-weighted vs random acquisition loop
  amortization.py      surrogate vs oracle wall-clock analysis
  hard_regime.py       low-viscosity shock evaluation
  calibrate.py         uncertainty calibration (reliability, OOD AUROC)
  stats_seeds.py       multi-seed statistics (full protocol, slow)
  stats_seeds_quick.py multi-seed statistics (quick mode)
  train_full.py        full training script (10k pairs, B 25ep / C 15ep)
  train_blind.py       train blind predictor V
  eval_full.py         full evaluation → RESULTS.json
  data_full.npz        10,000×(2,64,64) training pairs
  weights_B.pt         trained B; weights_C0/1/2.pt  trained ensemble
  weights_V.pt         trained blind predictor V
  RESULTS.json/.md     numbers + interpretation; train_log.json  provenance
```

## 8. Run

From `virtual-labs/` (so `import virtual_lab` resolves):

```bash
python -c "import virtual_lab; print('ok')"
python -m pytest tests/ -q                                    # contract tests (10)
python -m virtual_lab.domains.fluids.benchmarks --quick       # smoke Tests 1–8
python -m virtual_lab.domains.fluids.agent_example --quick    # CEM planner demo
python -m virtual_lab.domains.fluids.planner --quick          # CEM vs random
python -m virtual_lab.domains.fluids.active_demo --quick      # active loop demo
python -m virtual_lab.domains.fluids.calibrate                # uncertainty calibration
python -m virtual_lab.domains.fluids.hard_regime              # low-viscosity test
python -m virtual_lab.domains.fluids.amortization             # wall-clock analysis
python -m virtual_lab.domains.fluids.stats_seeds_quick        # multi-seed stats
python -m virtual_lab.domains.fluids.train_full               # full training
python -m virtual_lab.domains.fluids.train_blind              # train blind predictor
python -m virtual_lab.domains.fluids.eval_full                # full eval → RESULTS.json
```

## 9. Limits & next steps

- **Burgers, not Navier–Stokes; 64×64; Gaussian-force action space.** The NS
  vorticity spectral flag, richer actions (boundary velocity, geometry), and
  conservation/symmetry-constrained hybrids are the obvious upgrades.
- **Horizon H=13 at eps=1e-3.** Honest trust radius, not a failure — the lab
  re-grounds on the oracle beyond it. Decision horizon (T6b) is much shorter
  (H=1 for some seeds), meaning MSE-horizons overstate decision validity.
- **OOD AUROC=0.500.** Three near-identical seeds produce insufficient
  diversity for obstacle-OOD detection via ensemble variance. Fix: dropout MC,
  disagreement-promoting training, or explicit OOD detectors.
- **Active loop: uncertainty-weighted ≈ random.** Ensemble variance is uniform
  across candidates (~3e-3), so the uncertainty bonus doesn't re-rank. Need
  diverse ensemble members or explicit OOD detectors for informative
  uncertainty.
- **Amortization is negative for this toy oracle.** The CPU Burgers solver is
  17× faster than the surrogate. The surrogate's value is enabling virtual
  screening (10k+ candidates), not speed. For expensive oracles (3D CFD,
  wet-lab), the amortization argument becomes the core value proposition.
- **Hard regime is mild.** 64×64 can't resolve shocks at nu=0.001, so error
  only increases 26%. Higher resolution would expose the real failure.
- **Roadmap:** explicit state → latent state `z_t = E(s_≤t)` → pixels-only
  (the Veo test); query-conditioned abstraction (simplest representation
  sufficient for the intervention query); then the same
  state + intervention → trajectory + uncertainty + branching + active
  validation abstraction ported to materials, chemistry, robotics domains.
