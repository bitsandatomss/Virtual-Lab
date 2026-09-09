# Virtual Lab Vision: Quarantine Philosophy

## The core distinction: surrogate ≠ simulation

A **simulation** explicitly specifies mechanism: Navier-Stokes discretized,
material constants measured, boundary conditions encoded. You can inspect
every line of the PDE solver and know what it will do.

A **surrogate** is an empirical behavioral approximation: a neural network
trained on interventional data `P(Y|do(A))`. It preserves the input-output
mapping under the interventions it was trained on, but its internal
representation is opaque. You cannot inspect "the physics" inside a UNet.

This is not a flaw — it is the **fundamental epistemic position** of the
project. We do not claim to have learned the true dynamics. We claim to
have learned a function that is **experimentally sufficient** for a defined
class of interventions.

## What "experimental sufficiency" means

A surrogate is experimentally sufficient when:

1. **One-step fidelity** (T1): `error(Ŝ(s,a), S'(s,a)) < ε` for the
   intervention class you care about.
2. **Rollout stability** (T2): errors do not compound catastrophically over
   the validated horizon `H_ε`.
3. **Counterfactual validity** (T8): changing the action changes the
   predicted outcome in the same direction and magnitude as reality.
   Blind models (no action input) fail this test — they predict identical
   futures for different actions.
4. **Decision equivalence** (T6b): the surrogate's top-ranked candidate
   matches the oracle's top-ranked candidate for the same objective.

What is *not* required:
- Global state reconstruction accuracy
- Mechanistic interpretability
- Generalization to intervention classes never seen in training
- Physically meaningful latent representations

## The quarantine

Claims about the surrogate are quarantined from claims about reality by
three boundaries:

### 1. Validated horizon `H_ε`

The surrogate is trustworthy only up to `H_ε` steps, where `H_ε` is the
largest horizon such that the surrogate's ranking of candidate actions
matches the oracle's ranking. Beyond `H_ε`, the surrogate is extrapolating
and its outputs are hypotheses, not predictions.

`H_ε` is measured, not assumed. It depends on:
- The intervention class (some actions are harder to predict)
- The viscosity regime (low-viscosity shocks degrade faster)
- The model capacity and training data coverage

### 2. Oracle as validator, not oracle as crutch

The oracle (real experiment, high-fidelity simulation, or ground-truth
physics) serves two roles:
- **Training signal**: interventional trajectories for fitting
- **Validation gate**: the final arbiter of whether a surrogate claim holds

The surrogate is never evaluated against itself. Every `Δ_V2R`,
`R_discovery`, and `H_ε` measurement involves an oracle call. The value
proposition is: 10,000 virtual screens → 10 oracle validations, instead of
10,000 oracle calls.

### 3. Honest failure modes

The surrogate fails predictably and measurably:

| Failure mode | How detected | Example |
|---|---|---|
| OOD intervention | Variance spike (ensemble) or high error on held-out amp/nu | Force amp=3.0 outside training range [-2,2] |
| Horizon expiry | `H_ε` exceeded; ranking diverges from oracle | CEM plan valid for 5 steps, not 10 |
| Structural blindness | Blind model (V) fails T8 counterfactual test | Same state, different actions → identical prediction |
| Resolution limit | 64×64 can't resolve shock structures at low ν | Hard regime error increase is mild (26%) |

## What the surrogate enables that the oracle cannot

1. **Virtual screening**: 10,000 candidate action sequences evaluated in
   seconds, not hours. The oracle is too expensive for brute-force search;
   the surrogate makes exploration feasible.

2. **Counterfactual branching**: "What would have happened if I had pushed
   harder?" The oracle gives you one reality; the surrogate gives you all
   the roads not taken.

3. **Active learning**: The surrogate's uncertainty (or the blind model's
   failure) tells you where to query reality next. This closes the loop:
   surrogate proposes → oracle validates → surrogate improves.

4. **Amortized planning**: CEM in the surrogate finds near-optimal action
   sequences in 12s that would require 1000s of oracle calls. The training
   cost (~2 min) is amortized over many planning queries.

## What we are NOT claiming

- We are NOT claiming the surrogate learns the Navier-Stokes equations.
- We are NOT claiming the surrogate generalizes to all viscosities.
- We are NOT claiming the surrogate is faster than the oracle (on this
  CPU toy, the oracle is 17× faster per step).
- We are NOT claiming the ensemble captures all sources of uncertainty
  (OOD AUROC=0.500 — variance doesn't separate obstacle-OOD).

We ARE claiming: for the defined intervention class and validated horizon,
the surrogate is experimentally sufficient — it preserves the decision-
relevant behavior that matters for virtual experimentation.
