# fno-fox-infections
Repository for forward and inverse modeling of fox infections

This repository implements a **forward** time-conditioned
1D Fourier neural operator (FNO) trained on admissible source–target pairs
from a verified finite-volume reference solver, and a **noiseless full-field
inverse** pipeline for recovering diffusivity and transmission rate.

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

### 5. Identifiability-first inverse experiment

After training, run:

```bash
python -m fno_fox.inverse --config conf/default.yaml --data data/fox_rabies --checkpoint runs/default/best.pt --output runs/default/inverse
```

The default pilot selects ten test trajectories reproducibly (selection seed
0), five with constant initial susceptible fields and five with cosine fields
when available. Smaller test splits use all available cases. Selection never
uses recovery results. To choose cases explicitly, append `--sim-ids 1 17 42`
(each ID must be in the test split); use `--all-test` for the complete split.
`--device cpu` or `--device cuda` overrides automatic device selection.

The observations are both **complete S and I fields at every saved positive
time**, with the noiseless initial fields supplied as known inputs. The FNO is
frozen and each time is queried directly from t=0. Only `(D, beta)` is fitted;
mortality and initial-profile parameters remain fixed. The objective is half
the mean squared residual after the checkpoint's per-channel normalization,
with equal weights for all channels, times, and cells. Bounds come from the
checkpoint's conditioning ranges. No measurement noise is generated.

The inverse run first computes a 25×25 loss grid over the closed parameter box,
then computes local sensitivities at its minimum and at the known parameters.
It refines recovery with eight reproducible starts, including the grid minimum,
and reports the best result from the first four alongside the best from all
eight. The first four are reused, not rerun. Each start has at most 100
Nelder–Mead iterations and 250 objective evaluations. Four additional scalar
edge searches (at most 50 evaluations each) and explicit corners address
boundary solutions. Both comparisons share those candidates. Their reported
costs each include the grid and edges, so **do not add the arm totals**.
Budget exhaustion is recorded as nonconvergence, not successful optimization.
The best finite candidate is retained even if convergence was not established.

Sensitivities use two Jacobian–vector products through the FNO. Artifacts retain
physical derivatives with respect to D and beta, plus a dimensionless Jacobian:
outputs divided by checkpoint channel standard deviations, parameter columns
multiplied by their admissible ranges, and all entries divided by sqrt(number
of observed scalars). Diagnostics report column norms, signed column cosine,
singular values, condition number, and the weakest parameter direction.
Rank-deficient condition numbers and undefined zero-column cosines are JSON
`null`, with explicit rank/zero-column flags. These are **local FNO diagnostics**;
they do not establish structural identifiability of the PDE. Small column
norms and nearly parallel columns describe different sources of instability.

Validation residuals at true parameters retain signed bias and spatial/time
structure. Validation also selected the checkpoint during training, so these
are descriptive discrepancies, not independent uncertainty calibration.
There is no correlation multiplier, Gaussian likelihood, chi-squared threshold,
confidence interval, or bias correction. One-dimensional curves extracted
from the grid are labeled **coarse grid minima**.

Every recovered estimate is replayed with the finite-volume solver using the
known initial-profile specification. Identical four/eight-start estimates
reuse the replay and sensitivity calculation. No FV parameter optimization
is performed. Reported metrics include parameter signed/absolute/relative
errors and RMSE; S/I reconstruction, FV-versus-observation and FNO-versus-FV
RMSE and relative L2, overall and by time. Across-case distributions use median,
Q1, Q3, and IQR. Additional-start changes are flagged above 0.5% of either
parameter range, or a loss improvement above max(1e-10, 1% of four-start loss).
These configurable thresholds describe stability, not guaranteed accuracy.

Artifacts in the output directory:

- `manifest.json`: selected IDs written before inference, input hashes,
  checkpoint epoch, full configuration, versions, bounds, normalization, and
  completion/failure status.
- `cases.csv`, `summary.json`: both recovery estimates and aggregate metrics.
- `validation_residuals.npz/.json`: signed physical residual arrays and summaries.
- `cases/<sim_id>/`: loss grid, physical/scaled Jacobians, optimizer evaluation
  trace and termination records, observed/reconstructed fields, sensitivities,
  and per-time metrics in NPZ/CSV/JSON files.
- Recovery, start-comparison, sensitivity, validation-residual, and per-case
  joint-loss/reconstruction PNG figures.

Settings are in the `inverse` section of `conf/default.yaml`; old configs,
datasets, and checkpoints remain loadable without this section. The command
requires matching dataset and checkpoint provenance. Existing outputs are
protected unless `--overwrite` is supplied, and overwrite requires a previous
inverse manifest. Only IDs listed in the current manifest belong to that run;
older, unselected case directories may remain. Numerical case failures are
recorded and processing continues, with a nonzero CLI exit status if any fail.

### 6. Paired spatial-sensor inverse sweep

```bash
python -m fno_fox.inverse_sensor_sweep --config conf/default.yaml --data data/fox_rabies --checkpoint runs/default/best.pt --output runs/default/sensor_sweep
```

This entry point runs the same unique test trajectories through five noiseless
observation configurations. Both complete initial fields remain known; only
the later-time observations used for fitting are selected at sensors.

| Configuration | Spatial sensors | Positive observation times | Channels |
|---|---:|---:|---|
| Baseline | all 256 learning cells | 20 | S, I |
| Sparse A | 32 | 20 | S, I |
| Sparse B | 16 | 20 | S, I |
| Sparse C | 8 | 20 | S, I |
| Sparse D | 4 | 20 | S, I |

The sweep requires **100 unique test trajectories** by default
(`inverse.sweep_cases: 100`). It selects them reproducibly, aiming for 50 cases
from each initial susceptible-field family and filling from remaining test
cases if a family is too small. Every configuration uses the same selected IDs.
The existing 600-trajectory dataset has **only 90 test cases**, so the default
command above fails before model loading or inference. Supply a matching
dataset/checkpoint with at least 100 test cases to run the default protocol.
The sweep never repeats trajectories or borrows training/validation cases.
The standalone inverse command retains `inverse.pilot_cases: 10`.

To use an existing trained checkpoint with 100 **fresh** test trajectories,
append a verified test-only extension. This retains every original field,
parameter, split ID, and training normalization; the checkpoint is unchanged.
The inverse loader verifies the extension against the complete original
dataset and the checkpoint's original dataset hashes. The added trajectories
use the same physics, solver, parameter ranges, and grid, with a new sampling
seed and an exactly balanced initial-field family assignment.

```bash
python -m fno_fox.extend_test --data data/fox_rabies --output data/fox_rabies_test100 --count 100 --seed 6
```

For the original 600-case dataset, the new test IDs are 600 through 699.
Run only these IDs; the extension also retains the original 90 test cases.
On PowerShell:

```powershell
$newTestIds = 600..699
python -m fno_fox.inverse_sensor_sweep --config data/fox_rabies_test100/config.yaml --data data/fox_rabies_test100 --checkpoint runs/default/best.pt --output runs/default/sensor_sweep_100 --device cuda --sim-ids $newTestIds
```

The extension manifest records its source dataset, hashes, new IDs and
sampling seed. Both inverse manifests retain the checkpoint's training
dataset provenance separately from the extended evaluation dataset.

`--sim-ids` must provide exactly `inverse.sweep_cases` distinct test IDs.
`--all-test` runs the complete test split, provided it meets that minimum.
`--device` and `--overwrite` follow the standalone command's conventions;
overwrite requires an existing sensor-sweep manifest. Reducing `sweep_cases`
in a configuration explicitly changes the protocol and is useful for small
test fixtures. The baseline uses all cells on smaller grids; at least 32
learning cells are required. All saved positive times are used, so smaller
fixtures also have fewer observation times.

Sensors are learning-grid cell indices
`rint(linspace(0, Nx-1, n_sensors))`, approximately equally spaced from the
first through last cell centre. Their coordinates, channels, and times are
fixed across trajectories. No interpolation or noise is added. The layouts
are independently uniform for each count; they are not nested subsets.
The FNO still predicts on the complete grid. Fitting losses and observation
Jacobians use only the selected cells, with the existing global channel
normalization. Each selected Jacobian uses
`m = 2 * sensor_count * positive_time_count` for its `1/sqrt(m)` scaling;
`observed_scalars` is saved explicitly. These are average dimensionless
sensitivities, not confidence or information estimates.

Each configuration retains the 25×25 loss grid, four/eight-start comparison,
optimizer budgets, corner checks, boundary searches, and sensitivity
diagnostics at truth, the grid minimum, and both recovered estimates. The
seven random starts are identical across configurations for each trajectory;
the first start is each configuration's own grid minimum. Parameter bounds,
normalization, channels, times, model, and case IDs stay fixed.

Grid predictions are computed once per trajectory and reduced to five
sensor-specific losses. Validation predictions are also computed once and
selected for each layout. Identical diagnostic parameter pairs reuse full
predictions, Jacobians, and FV replays within a trajectory. No FV parameter
optimization, probabilistic error model, or uncertainty intervals are used.
Repeated optimizer queries use a per-trajectory cache capped at 64 full
predictions. Actual forward queries and cache hits are recorded separately
from optimizer objective-evaluation counts.

Reports retain parameter signed/absolute/relative errors and RMSE, plus
physical S/I bias, RMSE, and relative L2 overall and by time. FNO-versus-data,
FV-versus-data, and FNO-versus-FV metrics each have explicit **sensor** and
**full_grid** scopes. Full-grid reference fields are used for evaluation
after fitting; unobserved later-time cells do not enter the fitting loss.
Across-case summaries use median, Q1, Q3, and IQR. Paired changes are sparse
minus baseline, matching both simulation ID and four/eight-start count.
Successful and failed counts and the paired sample sizes are reported.

The output contains:

- `manifest.json`: the selected IDs and layouts written before inference,
  provenance, times/channels, normalization, bounds, and completion status.
- `cases.csv`, `summary.json`, `paired_changes.csv`: configuration/start-count
  results, aggregates, and paired changes relative to baseline.
- `shared_costs.json`: shared grid, validation, diagnostic prediction,
  Jacobian, and FV computation. Configuration optimizer totals exclude the
  shared grids; four/eight-start totals must not be added together.
- `validation_residuals.npz/.json`: shared descriptive full-grid residuals.
- `configurations/<name>/`: selected validation residuals, configuration
  summaries and figures, and `cases/<sim_id>/` numerical artifacts and plots.
- `sensor_recovery.png`, `sensor_replay_errors.png`,
  `sensor_sensitivities.png`, `sensor_start_stability.png`: comparisons versus
  sensor count. The first three use eight-start results; error bars show the
  interquartile spread across successful trajectories, not confidence bounds.

Numerical failures are retained per configuration and trajectory, processing
continues, and any failure gives a nonzero CLI exit status. Paired summaries
only use matching successful cases and explicitly count exclusions. On
overwrite, only IDs listed in the current manifest belong to the new run;
older unselected case directories may remain.

### 7. Paired spatial-and-temporal inverse sweep

```bash
python -m fno_fox.inverse_sensor_time_sweep --config conf/default.yaml --data data/fox_rabies --checkpoint runs/default/best.pt --output runs/default/sensor_time_sweep
```

This command combines **16, 8, and 4 spatial sensors** with **20, 15, 10, 7,
and 5 positive observation times**, giving 15 configurations and 1,500
recovery problems for the same 100 unique test trajectories. Every problem
retains the four/eight-start comparison; the first four starts are reused,
so those comparisons do not double the optimization work. All configurations
are computed afresh. There is no import of previous sweep results.

`inverse.sweep_time_counts: [20, 15, 10, 7, 5]` controls the temporal counts;
the three spatial counts are fixed. Existing configuration files and
checkpoints acquire these defaults without regeneration. Existing commands
retain their observation-selection behavior. Case selection and the
`--sim-ids`, `--all-test`, `--device`, and `--overwrite` arguments follow the
spatial sweep. Overwrite requires a manifest from this new entry point.
The current 90-case test split still fails the default 100-case requirement
before model loading or inference. This command also requires at least 16
learning cells and as many positive snapshots as the largest requested count.
Counts must be distinct integers of at least two; they are never reduced
silently.

Use `--max-recovery-problems 500` to cap the number of case/layout recoveries
while retaining both start arms. The schedule follows the selected trajectory
order, then the configuration order (16/8/4 sensors, descending time counts).
With `inverse.sweep_cases: 34`, this runs all 15 configurations for the first
33 trajectories (495 problems), followed by all five time counts at 16 sensors
for the 34th trajectory. Counts are 34 per 16-sensor configuration and 33 per
8/4-sensor configuration. The manifest records scheduled IDs per configuration
and budget exclusions; exclusions are distinct from failed recoveries.

`--resume` reuses complete per-case artifacts from an interrupted run of this
entry point. It verifies the checkpoint and dataset hashes, normalization,
observation layouts, device, and recovery settings before writing output. The
selected cohort can shrink to a subset of the original IDs, and the cap can
change; other scientific settings must match. Saved failures remain recorded.
Partial cases without a success marker are recomputed. `--resume` and
`--overwrite` are mutually exclusive. For the 500-problem run in this workspace:

```powershell
$testIds = 600..633
python -m fno_fox.inverse_sensor_time_sweep --config runs/default/sensor_time_sweep_500_config.yaml --data data/fox_rabies_test100 --checkpoint runs/default/best.pt --output runs/default/sensor_time_sweep_500 --device cuda --sim-ids $testIds --max-recovery-problems 500 --resume
```

Spatial layouts match the existing sweep. Temporal layouts are approximately
uniform **nested subsets of saved positive times**. The smallest count starts
with `rint(linspace(0, N_positive-1, count))`. Additional observations split
the largest interval containing unused snapshots, choosing the saved time
nearest its midpoint. Ties choose the earlier interval or timestamp, allowing
for floating-point roundoff. This gives
`T5 ⊂ T7 ⊂ T10 ⊂ T15 ⊂ T20`, with the first and last positive times included.
The 20-time configurations use all positive snapshots in the default dataset,
and therefore match the corresponding original 16/8/4-sensor experiments
when IDs, checkpoint, and optimizer settings match. If a dataset has more
than 20 positive snapshots, 20 denotes a selected subset instead.

No noise or temporal interpolation is introduced. Both complete initial
S/I fields remain known; each prediction is queried directly from those
fields. Only selected cells and times enter fitting or observation Jacobians.
The scaled Jacobian uses `m = 2 * sensor_count * time_count`, recorded as
`observed_scalars`. Joint grids, validation predictions, full Jacobians,
and repeated diagnostic/FV calculations are shared within a trajectory.
Cached forward outputs span every saved positive time, so actual query cost
does not scale solely with the fitted time count. Each configuration still
has its own loss surface, grid-minimum first start, optimization, and recovery.

Both start counts report all three field comparisons, with physical S/I bias,
RMSE, and relative L2 overall and by time, under these scopes:

| Scope | Evaluated values | Role |
|---|---|---|
| `sensor` | Selected cells at selected times | Fitting observations |
| `full_grid_selected_times` | All cells at selected times | Evaluation against ground truth, including spatially withheld values |
| `full_grid` | All cells at all saved positive times | Evaluation against ground truth, including spatially and temporally withheld values |

The complete reference fields in the last two scopes do not enter the fitting
loss. These scopes describe the domains of comparison, including when the
comparison is FNO-versus-FV. Parameter metrics, local sensitivities, solver
budgets, explicit boundary checks, and nonconvergence records are retained.
There are no confidence bounds or probabilistic discrepancy models.

The output retains `manifest.json`, `cases.csv`, `summary.json`,
`paired_changes.csv`, `shared_costs.json`, and shared validation artifacts.
The manifest saves IDs and all observation layouts before inference, including
actual timestamps and zero-based indices into the positive snapshots.
`configurations/sensors_<count>_times_<count>/cases/<sim_id>/` contains grids,
optimizer traces, full fields with selection masks, selected Jacobians,
metrics, and figures. Reconstruction plots mark fitted observations only at
fitted times and label other target fields as evaluation ground truth.
Root comparison figures are `sensor_time_recovery.png`,
`sensor_time_replay_errors_<scope>.png`, `sensor_time_sensitivities.png`, and
`sensor_time_start_stability.png`. They use time count on the horizontal axis
and separate spatial-layout curves; main results use eight starts, with
interquartile spread rather than confidence bounds.

Paired **temporal** changes compare against 20 times at the same sensor count
(the largest count if defaults are overridden). Paired **spatial** changes
compare against 16 sensors at the same time count. Differences are current
minus reference, matching simulation ID and start count. Reports explicitly
count pairs and exclusions, with nonconvergence separate from failed cases.
Failures preserve partial artifacts and give a nonzero CLI status. Older
unselected case directories may remain on overwrite; the current manifest
identifies the run's cases.

#### Optional two-trajectory benchmark

Before the production sweep, benchmark the same model and optimizer settings
on two trajectories selected from the intended production cohort. The
following preparation script reads the dataset and selects IDs; it performs
no inference. Use your matching dataset/configuration with at least 100 test
cases. It chooses one case from each family when both are available and fills
from the cohort otherwise:

```python
from dataclasses import replace
from pathlib import Path
import numpy as np
import yaml
from fno_fox.config import load_config
from fno_fox.data import load_dataset
from fno_fox.inverse_sensor_sweep import select_sweep_cases

config = load_config("conf/default.yaml")
bundle = load_dataset("data/fox_rabies")
cohort = select_sweep_cases(bundle, config.inverse)
family = bundle.parameters["susceptible_constant"][cohort].astype(bool)
chosen = [int(cohort[family == value][0]) for value in (True, False)
          if np.any(family == value)]
chosen.extend(int(sim) for sim in cohort if int(sim) not in chosen)
ids = chosen[:2]
benchmark = replace(config, inverse=replace(config.inverse, sweep_cases=2))
Path("conf/sensor_time_benchmark.yaml").write_text(yaml.safe_dump(benchmark.as_dict()))
print("Benchmark --sim-ids:", *ids)
```

Run the benchmark explicitly, replacing the IDs below with those printed:

```bash
python -m fno_fox.inverse_sensor_time_sweep --config conf/sensor_time_benchmark.yaml --data data/fox_rabies --checkpoint runs/default/best.pt --output runs/default/sensor_time_benchmark --sim-ids <first-id> <second-id>
```

This runs 30 recovery problems with all production optimization budgets and
never launches the larger experiment automatically. `summary.json` records
total wall time; `shared_costs.json` records validation time and each
trajectory's `wall_seconds_including_grid`. A rough production estimate is
the one-time validation cost plus 100 times the benchmark's mean trajectory
time, with figure generation and reporting overhead added separately.
Trajectory difficulty and cache reuse can change this estimate. Eight-start
arm costs already contain the first four starts and boundary searches:
**do not add four-start and eight-start totals** or count shared grid costs
once per configuration.

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
- **Inverse:** exact full-field observations, provenance and selection,
  synthetic recovery (interior/edge/corner optima, narrow valleys, multiple
  basins), hard evaluation budgets, nested-start reuse, FFT Jacobians versus
  finite differences, rank-deficient sensitivities, physical metrics, FV
  replay, and a tiny trained-checkpoint end-to-end test with saved figures.
- **Spatial-sensor sweep:** strict unique-case selection, uniform layouts,
  full-grid baseline equivalence, isolation from unobserved target cells,
  masked Jacobians, shared computation, paired metrics/failures, and a tiny
  all-five-configuration end-to-end fixture.
- **Spatial-and-temporal sweep:** nested saved-time selections, the 20-time
  baseline, isolation from unobserved cells and times, selected Jacobian
  scaling, three evaluation scopes, paired references along both axes,
  shared caches, early validation, and a tiny all-15-configuration fixture.

The pilot refinement test is marked `slow`; deselect it with `-m "not slow"`.

## Scope

The forward experiment reports held-out FNO performance against persistence,
separately for the two source categories. It makes no claim about accuracy
targets or speedup. The inverse experiment reports deterministic recovery and
local surrogate identifiability diagnostics without probabilistic intervals.
This repository deliberately excludes autoregressive rollout studies, semigroup-consistency
losses, PINNs, additional architectures, DDP or cluster tooling, OOD suites,
and hyperparameter sweeps.

Copied code and its provenance (source revisions and file hashes) are listed
in [SOURCES.md](SOURCES.md). The model brief is *Learning the Spatial Spread
of Fox Rabies with a Fourier Neural Operator* (revised project brief,
October 2026).
