# Virtual Lab — Full Training Results

Device: `cuda` | seed 0 | n=64 | full data [10000, 2, 64, 64] (200 trajs x 50 steps = 10k pairs) | holdout [1600, 2, 64, 64] (seed 123) | OOD obstacle [320, 2, 64, 64].

## Training

| model | epochs | batch | base | seed(s) | final loss (MSE residual) |
|---|---|---|---|---|---|
| B (SurrogateB) | 25 | 64 | 32 | 0 | 1.352439e-05 |
| C member 0 | 15 | 64 | 32 | 0 | 2.511501e-05 |
| C member 1 | 15 | 64 | 32 | 1 | 1.753452e-05 |
| C member 2 | 15 | 64 | 32 | 2 | 2.263262e-05 |

Losses are one-step residual MSE (Y_next - Y_t); rollout/ood errors below are state MSE (u,v). Batched GPU inference is exactly equivalent to per-sample SurrogateB.predict for these protocols.

## Tests 1-7 (trained models, no retraining)

| test | A (baseline) | B | C (ensemble mean) |
|---|---|---|---|
| T1 one-step MSE | - | 2.152164e-06 | 1.716536e-06 |
| T2 rollout-20 mean MSE | 1.235401e-02 | 9.708792e-04 | 8.952717e-04 |
| T3 interventional holdout MSE | - | 4.919730e-05 | 6.769324e-05 |
| T4 obstacle-OOD MSE | - | 7.455396e-05 | 7.398534e-05 |
| T4 ensemble variance (OOD / train) | - | - | 2.179116e-06 / 1.897797e-06 |
| T5 nu=0.05 rollout-20 mean MSE | - | 5.071909e-04 | 3.782975e-04 |
| T6 validated horizon H(eps=1e-3, 20 steps) | - | 13 | 13 |
| T6 extended H(eps=1e-3, 50 steps) | - | 13 | 13 |
| T7 discovery R (top-5/50) | - | 0.400 | 0.800 |
| T7 Delta_V2R (virt-real energy) | - | 1.066215e-03 | -9.300797e-04 |

## Agent demo (10000 virtual -> top-10 oracle validation)

| n_virtual | topk | virt_best (enstrophy) | real_best (enstrophy) | Delta_V2R | R_discovery |
|---|---|---|---|---|---|
| 10000 | 10 | 5.013226e-01 | 4.986517e-01 | 2.982961e-03 | 0.600 |

## V2R interpretation

- **Delta_V2R** = mean(virtual - real) score of the selected candidates. Positive = surrogate over-optimism (virtual screening promises more than the oracle delivers); negative = pessimism. Small |Delta_V2R| with high R_discovery means the surrogate ranks like reality even if absolute scores shift.
- **R_discovery** = fraction of the oracle top-k recovered by virtual screening. It is the decision-relevant metric: a surrogate is useful if it surfaces the same interventions the oracle would.
- **H_eps (validated horizon)** = steps until surrogate/oracle MSE exceeds eps; the trust radius for multi-step virtual rollouts. Inside H, agents can plan freely; beyond it, predictions must be re-grounded on the oracle.
- **Ensemble C** adds uncertainty: member variance flags OOD actions (compare T4_var_C on obstacle-OOD vs train samples: higher OOD variance drives the ood flag), so the agent can trade off predicted score vs. variance (explore where uncertain, exploit where validated).

## Learned Environment vision

- **Reality -> Surrogate**: interventional data (random do(A) force / viscosity / obstacle sequences on the Burgers oracle, 10k pairs) trains SurrogateB/C to emulate one-step dynamics.
- **Surrogate -> Agents**: agents screen 10,000 candidate interventions entirely inside the learned environment (batched GPU rollouts) and nominate only the top-10 for reality — a 1000x reduction in oracle queries per discovery round.
- **Agents -> Reality (closing the loop)**: the top-k are validated on the oracle; Delta_V2R/R_discovery quantify the virtual-to-real gap and decide what gets re-added to training data (active_loop.propose / validate). Repeated rounds shrink the gap where it matters.
- **Experimental sufficiency**: the suite (one-step error, rollout error, holdout, obstacle-OOD, nu-extrapolation, validated horizon, discovery rate, V2R gap) specifies *when the surrogate is sufficient* for an experiment: rank-preserving (high R_discovery) inside H_eps with bounded |Delta_V2R|. Outside that envelope the lab demands fresh oracle experiments instead of trusting the virtual lab.
