# fno-fox-infections
Repository for forward and inverse modeling of fox infections

This repository currently implements the **forward** part: a time-conditioned
1D Fourier neural operator (FNO) trained on admissible source–target pairs
from a verified finite-volume reference solver. Inverse inference is not
implemented yet.

## Model and operator

The minimal fox-rabies reaction–diffusion system on `x ∈ [0, L]` with zero
infectious flux at both ends:

```
S_t = -β S I
I_t = D I_xx + β S I - μ I
```

The FNO learns the evolution operator

```
G_θ(S(·, t_s), I(·, t_s), D, β, Δt) ≈ (S(·, t_j), I(·, t_j)),   Δt = t_j - t_s > 0
```

The PDE is autonomous, so absolute source time is not an input. Both source
density fields are. Defaults: `L = 10`, `T = 5`, `μ = 0.3`, 256 learning-grid
cells, 21 snapshots (including `t = 0`), and 600 trajectories split
420/90/90 by complete trajectory.

## Setup

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -e ".[test]"
```

Dependencies are pinned in `pyproject.toml` (numpy 2.2.6, scipy 1.15.3,
torch 2.7.0, PyYAML 6.0.2, matplotlib 3.10.8; Python ≥ 3.11). Training uses
one CUDA GPU when available and the CPU otherwise.

## Pipeline

Run the four entry points in order. Every path is explicit.

```bash
python -m fno_fox.verification --config conf/default.yaml --output runs/verification/report.json
```

```bash
python -m fno_fox.generate --config conf/default.yaml --output data/fox_rabies
```

```bash
python -m fno_fox.train --config conf/default.yaml --data data/fox_rabies --run runs/default
```

```bash
python -m fno_fox.evaluate --config conf/default.yaml --data data/fox_rabies --checkpoint runs/default/best.pt --output runs/default/evaluation
```

`generate` and `train` refuse to overwrite existing outputs unless you pass
`--overwrite`. `train` also accepts `--epochs` and `--device` (`auto`, `cpu`,
`cuda`, ...). All settings live in [conf/default.yaml](conf/default.yaml).

### 1. Reference solver and verification

[`physics.py`](src/fno_fox/physics.py) uses cell-centred finite volumes with a
conservative zero-flux Laplacian. Time stepping is coupled Crank–Nicolson for
`(S, I)`, solved in float64 with an analytic sparse Jacobian and damped Newton
iterations with Armijo backtracking. A step converges only when both the
residual and the update are below tolerance. Failed convergence, an exhausted
line search, or non-finite values raise an error with the step, time, norms,
and worst cell. Trajectories are never clipped, and failed cases are never
replaced. Each save interval is subdivided so steps land exactly on the
requested save times.

The solver grid is `refinement_factor × 256` cells (default 512) with
`dt = 0.01`. Solutions are conservatively averaged onto the 256-cell learning
grid. Initial profiles are exact cell averages, so every grid sees the same
initial data.

`python -m fno_fox.verification` writes a JSON report with these checks:

| Check | Criterion |
|---|---|
| Diffusion-only analytic solution (β = 0) | spatial order ≥ 1.8 |
| Uniform reaction vs. DOP853 ODE reference | temporal order ≥ 1.8 |
| Zero-infection equilibrium | exact fixed point |
| Coupled manufactured solution | spatial and temporal order ≥ 1.8 |
| Population balance `d/dt ∫(S+I) = -μ ∫I`, using the solver's trapezoidal mortality term | relative residual ≤ 1e-10 |
| Analytic Jacobian vs. central finite differences | relative difference < 1e-8 |
| Positivity and finite values on corner pilots | min ≥ -1e-10 |
| Failure modes (iteration limit, failed backtracking, non-finite input) | informative error raised |
| Pilot refinement (2× cells and dt/2, separately) | channel-wise normalized RMS change < 0.5% |

The refinement change is the per-pilot learning-grid RMS difference divided by
the channel's standard deviation pooled over all pilots. At the defaults,
observed orders are about 2.0 (lowest 1.96). Refinement changes are about
1e-4 in space and 1e-5 in time, far below the 0.5% target.

### 2. Data generation

[`sampling.py`](src/fno_fox/sampling.py) draws:

- `(D, β)` from a Latin hypercube on `[0.02, 0.08] × [0.6, 1.2]`, with no
  imposed correlation.
- `I₀ = A exp(-(x - x_c)² / (2σ²))`, with `A ∈ [0.01, 0.05]`,
  `x_c ∈ [4.5, 5.5]`, and `σ ∈ [0.2, 0.4]` each uniform and independent.
- `S₀`: an exact 50/50 mix of `S₀ = 1` and
  `S₀ = 1 + a Σ_{k=1..4} c_k cos(kπx/L)`, with `a ∈ [0.05, 0.2]` and normal
  `c_k` normalized to `Σ|c_k| = 1`. This keeps `S₀ ≥ 1 - a > 0`.

`generate` first runs the pilot refinement study and aborts if it exceeds the
threshold. It then solves every trajectory, checks each one for finite values,
positivity, and balance (aborting on any failure), splits complete
trajectories, and writes two files:

- `dataset.npz` (compressed): the learning-grid `x_centers`, `times`,
  `susceptible`/`infectious` arrays `[n, 21, 256]` (float32), every sampled
  parameter including the `S₀` coefficients, and the split IDs.
- `manifest.json`: the configuration and its hash, seeds, grid and time
  coordinates, splits, normalization, per-trajectory parameters and solver
  diagnostics, the refinement report, the dataset hash, and the code's git
  state.

Normalization is a per-channel mean and standard deviation over every training
trajectory, snapshot, and cell. It is not weighted by how often a snapshot
appears in pairs, and `load_dataset` recomputes and checks it.
Loading also verifies the dataset file's SHA-256 against the manifest before
reading the arrays.

On this machine, the default 600-trajectory dataset took about 110 s to
generate, single-threaded. Every step converged in at most two Newton
iterations with no backtracking. The largest relative balance residual was
4e-15, and the most negative I was -6e-18.

### 3. Admissible source–target pairs and training

A pair `(sim, s, j)` always takes both snapshots from the same trajectory,
with `0 ≤ s < j ≤ 20`. Pairs are stored as index triples. Each sample provides
the source fields, `(D, β, Δt)`, the target fields, the simulation ID, and the
source and target indices.

Each training epoch uses exactly 40 pairs per training trajectory (16,800
examples):

- all 20 initial-source pairs `(0, j)`;
- 20 later-source pairs, drawn with replacement: the lag is uniform on
  `1…19`, then the source is uniform on `1…20 − lag`.

Every admissible later-source pair can be drawn, and lead times are
balanced. The later-source draw is refreshed each epoch from the seed and the
epoch number. Validation uses the same 20 + 20 structure, with the later
pairs drawn once and then frozen. Pair count does not add independent
parameter realizations: simulation splits stay disjoint.

Model inputs, using the interface `model(spatial, cond) → prediction` from the
source FNO:

- `spatial [B, 256, 3]`: normalized source `S`, normalized source `I`, and `x/L`;
- `cond [B, 3]`: range-scaled `D`, range-scaled `β`, and `Δt/T`;
- output `[B, 256, 2]`: normalized target `S` and `I`.

Normalization is global per channel and never per sample, so absolute density
and depletion remain visible to the model. The architecture uses 16 modes,
width 32, four Fourier blocks, CIN conditioning with hidden width 64, and no
dropout.

Training uses equal-channel normalized MSE, AdamW (lr `1e-3`, weight decay
`1e-4`), batch size 64, and cosine annealing over 200 epochs, in float32 with
`num_workers = 0`. The best checkpoint minimizes the equally weighted mean of
the initial-source and later-source validation losses. The run directory
holds `best.pt`, `last.pt`, `config.json`, `normalization.json`,
`metrics.csv` (training loss plus separate initial-source and later-source
validation losses per epoch), and `summary.json`. On a 4-thread CPU an epoch
takes about 18 s.

Training on later-source pairs does not require autoregressive inference.
Every query is answered directly, so an inverse method can still query every
observation time from known initial fields.

### 4. Evaluation

`evaluate` queries all 210 admissible pairs of every held-out trajectory and
writes:

- `metrics.json`: S and I RMSE and relative L2 in physical units, reported
  separately for initial-source and later-source queries and for each lead
  time. Each metric is reported for the FNO and for **source persistence**
  (the source `(S, I)` returned unchanged). The file also holds negative
  prediction counts and minimums (predictions are not clipped), front
  diagnostics, and the interpolation check. Non-finite predictions abort
  evaluation; unavailable diagnostics are written as JSON `null`.
  A relative L2 is undefined, and
  counted separately, when the reference RMS is ≤ 1e-12.
- `pair_metrics.csv`, `fronts.csv`, `front_speeds.csv`.
- `profiles.png`: truth, prediction, and error profiles for three
  representative test trajectories, chosen at low, median, and high error.
- `error_vs_lead_time.png`: RMSE against lead time, FNO vs. persistence, for
  both source categories.
- `front_position.png`: initial-source front trajectories, plus fitted speed
  (FNO vs. truth).

The right front is the outermost downward crossing of `I = 0.01`, located by
linear interpolation. A front that is absent or reaches the boundary is
marked unavailable. Front speed is a least-squares fit over `t ∈ [2.5, 5]`
when at least four valid crossings exist. The asymptotic speed
`2√(D(β − μ))` is reported only as a diagnostic for uniform-background
cases.

**Continuous lead-time interpolation.** The first ten test trajectories are
re-solved with extra save times halfway between the stored snapshots, then
queried from `t_s = 0` and `t_s = 2.5`. Errors at these midpoint targets are
reported next to the errors at saved targets.
With `--split train` or `--split val`, both the main evaluation and interpolation
check use that requested split. Interpolation source times must match saved
snapshots strictly before the final snapshot; invalid settings fail before
model inference or interpolation reference solves.

## Tests

```bash
python -m pytest
```

The suite covers:

- **Physics:** the verification checks above, exact save-time landing,
  conservative averaging, and acceptance flags.
- **Data and pairs:** reproducible distributions, positive initial fields,
  disjoint splits, training-only statistics, same-trajectory forward pairs,
  lead times, retention of all initial-source pairs, lag balance and support,
  deterministic epoch resampling, and frozen validation pairs.
- **Model:** the tensor contract, gradients through CIN, padding removal, and
  an exact checkpoint round trip.
- **Pipeline:** a tiny CPU generate/train/evaluate run, a single-GPU smoke
  test (skipped without CUDA), and a small-data overfit check covering both
  initial and later sources.

The pilot refinement test is marked `slow`; deselect it with `-m "not slow"`.

## Scope

The forward experiment reports held-out FNO performance against persistence,
separately for the two source categories. It makes no claim about accuracy
targets, inverse identifiability, or speedup. This repository deliberately
excludes inverse scripts, autoregressive rollout studies, semigroup-consistency
losses, PINNs, additional architectures, DDP or cluster tooling, OOD suites,
and hyperparameter sweeps.

Copied code and its provenance (source revisions and file hashes) are listed
in [SOURCES.md](SOURCES.md). The model brief is *Learning the Spatial Spread
of Fox Rabies with a Fourier Neural Operator* (revised project brief,
October 2026).
