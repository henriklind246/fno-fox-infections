# Copied and adapted source code

Code was copied into this repository on 2026-10-12. There is no runtime
dependency on the source repositories and no heat-transfer model weights are
imported. Hashes are SHA-256 of the file contents.

## Source revisions

| Repository | HEAD at copy time | Working tree |
|---|---|---|
| `/Users/henriklind/projects/no-tps-ihcp` | `6f58b13bdc874dba0c3c8da3c3b1b1ec6749dea9` | copied files clean |
| `/Users/henriklind/projects/fno-tps-sizing` | `bcbaa10d6e74da9f2cb95f974b12828111efbcda` | one copied file locally modified (below) |

## Files

| Source | Destination (`src/fno_fox/` unless noted) | Working-tree SHA-256 | State |
|---|---|---|---|
| no-tps-ihcp `src/operators/fno1d.py` | `model.py` | `d5ee98b023107e27af0c66ade59694fc35db1fa6da10c870a5b3f68c9d1004b0` | clean |
| no-tps-ihcp `src/physics/init_conditions.py` | `sampling.py` | `a4dd4ccc2a2e4c63be2c207c1127feb633c2162ce1e48c5e2eb3b664a242c8a4` | clean |
| no-tps-ihcp `data/generate_dataset.py` | `sampling.py` | `6ad54a22c65aa35e3f99177c85274c5571e8f3092b2421e8fc9d10f1c240bdf3` | clean |
| no-tps-ihcp `data/dataset.py` | `data.py` | `516f4dad2cd0e5ba08720c465caa2691bb54ec58879731fe50eeff107e707837` | clean |
| fno-tps-sizing `src/fno_tps/physics.py` | `physics.py` | `17d7f1f5fb24715aefe6ccf0a35383027380c8759c194f93b302408e8a615ae0` | clean |
| fno-tps-sizing `src/fno_tps/verification.py` | `verification.py` | `4e1f9834be8326da26c8c747c0720a23409ce8e20ec736f573964053ac00a2ce` | clean |
| fno-tps-sizing `tests/test_physics_nonlinear.py` | `tests/test_physics.py` | `57a3b6ff258911e92eddc077b1238edeed10c9aa91fb1910b7f510f035f65b59` | clean |
| fno-tps-sizing `src/fno_tps/data.py` | `data.py`, `generate.py` | `df4b3fedc840ddbdec3f823e5183f93dee3c496ebff32b0995e34d580b516d90` | clean |
| fno-tps-sizing `src/fno_tps/training.py` | `train.py` | `3678462b92f8c39d55fbb89b5562a6063a303f5f8de1bb6ff086c00929779389` | clean |
| fno-tps-sizing `src/fno_tps/runtime.py` | `train.py`, `data.py` | `d848d7363eff3f83d71b1dd5eeeb267b7bf4b16d8a641e39cf26bafbdd418f2e` | clean |
| fno-tps-sizing `src/fno_tps/evaluation.py` | `evaluate.py` | `e0832667eca80f1f8cbbfe2f925110d64878a2f13ad135e1320e5f993c92b38f` | **modified** |

For clean files the working-tree hash equals the committed blob hash at HEAD.

### Local modification

`fno-tps-sizing/src/fno_tps/evaluation.py` differed from HEAD when copied
(committed blob `0d320c7af26e70e699372280816b225d5ee4fbd4cb38be9bafd257c0deb40246`;
`git diff HEAD` SHA-256 `21c264bde5eb586350a3bdde55037c33e96bcc6c70f249f81cedabca3ac3be23`,
100 insertions and 35 deletions). The uncommitted change replaces the scalar
safety buffer with `fno_tps.safety` constraint buffers and adds
underprediction-error fields. None of that safety logic was carried over;
`evaluate.py` takes only the batched-inference and physical-unit error
accumulation pattern, which is the same at HEAD and in the working tree.

## What was kept, changed, or removed

- `model.py`: spectral blocks, fixed padding of 8, conditioning MLP with
  identity initialization, and CIN copied verbatim; defaults changed to three
  spatial input channels, two outputs, a three-value condition, and conditioning
  width 64.
- `physics.py`: conservative cell-centred discretization, fixed sparse pattern
  with slot updates, and damped Newton with Armijo backtracking. The thermal
  residual and Jacobian are replaced by the coupled S/I Crank–Nicolson system. An
  exhausted line search now raises instead of accepting the last damped step.
- `verification.py`, `tests/test_physics.py`: manufactured-solution,
  convergence-order, balance, Jacobian, and failure-test patterns, applied to the
  S/I equations.
- `sampling.py`: analytic, grid-independent profile builders, Latin-hypercube
  scalars, and balanced family assignment, reduced to the 1D distributions.
- `data.py`: within-trajectory forward pairs, simulation-level splits,
  training-only normalization, and `set_epoch` resampling. Benchmark adapters,
  forcing encoders, curricula, and distributed samplers are removed.
- `train.py`: single-device loop, seeding, and best/last checkpoints. DDP,
  safety calibration, Pareto bookkeeping, and early stopping are removed.
- `evaluate.py`: batched inference; TPS margin and feasibility metrics replaced
  by S/I errors, persistence, and front diagnostics.
