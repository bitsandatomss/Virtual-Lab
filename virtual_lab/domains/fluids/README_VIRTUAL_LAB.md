# Virtual Lab skeleton

Learned-environment testbed on a 2D viscous Burgers oracle (numpy CPU, 64×64 periodic).

**Note:** `matplotlib` is NOT in `requirements.txt` (numpy/scipy/torch/tqdm are).
This package does not need it; install separately only for plotting.

## Layout

- `oracle.py` — Burgers oracle, `rasterize_force`
- `lab.py` — Lab API: reset/intervene/branch/rollout/measure/validated_horizon/ask_surrogate
- `data_gen.py` — random `do(A)` trajectories → `.npz` (`Y_t,A_t,Y_{t+1},C`)
- `models/model_a.py` — persistence + coarse baseline (no train)
- `models/model_b.py` — tiny Conv UNet one-step surrogate (~100k params, torch CPU)
- `models/model_c.py` — 3-ensemble of B with variance + OOD flag
- `benchmarks.py` — Tests 1–7
- `metrics_v2r.py` — `Delta_V2R`, `R_discovery`, `H_eps`
- `active_loop.py` — surrogate propose + oracle validate loop
- `agent_example.py` — 10k virtual → top10 validate demo

## Run

```bash
python -c "import virtual_lab; print('ok')"
python -m virtual_lab.benchmarks --quick
python -m virtual_lab.agent_example --quick
python -m virtual_lab.data_gen --quick --out virtual_lab/data_quick.npz
```
