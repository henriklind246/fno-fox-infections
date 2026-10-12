"""Generate the fox-rabies trajectory dataset.

    python -m fno_fox.generate --config conf/default.yaml --output data/fox_rabies

Adapted from fno-tps-sizing ``src/fno_tps/data.py`` (trajectory storage,
per-trajectory solver diagnostics, manifest, training-only statistics).
Sampled cases are never clipped or silently replaced: a trajectory that fails
its solve or acceptance checks aborts generation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import argparse
import json
import time

import numpy as np

from fno_fox.config import FoxConfig, load_config
from fno_fox.data import (
    DATASET_FILE,
    MANIFEST_FILE,
    SCHEMA_VERSION,
    capture_git_state,
    compute_normalization,
    file_sha256,
    split_simulation_ids,
)
from fno_fox.physics import Grid, NonlinearSolveError, acceptance_report, simulate
from fno_fox.sampling import sample_parameters
from fno_fox.verification import refinement_study


def generate_dataset(
    config: FoxConfig,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    verbose: bool = True,
) -> Path:
    destination = Path(output_dir)
    if (destination / DATASET_FILE).exists() and not overwrite:
        raise FileExistsError(
            f"{destination / DATASET_FILE} exists; pass --overwrite to replace it."
        )
    destination.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    refinement: dict[str, Any] | None = None
    if config.refinement.enabled:
        refinement = refinement_study(config)
        if verbose:
            print(
                "Refinement pilots: spatial "
                f"{refinement['spatial_max_normalized_rms_change']}, temporal "
                f"{refinement['temporal_max_normalized_rms_change']}"
            )
        if not refinement["passed"]:
            raise RuntimeError(
                "Pilot refinement exceeds the configured threshold "
                f"{config.refinement.threshold}; refine solver.refinement_factor "
                "or solver.dt before production generation."
            )

    parameters = sample_parameters(config.sampling, np.random.default_rng(config.seeds.sampling))
    n = len(parameters)
    times = np.asarray(config.domain.save_times, dtype=np.float64)
    cells = config.domain.learning_cells
    susceptible = np.empty((n, len(times), cells), dtype=np.float32)
    infectious = np.empty_like(susceptible)
    diagnostics: list[dict[str, Any]] = []
    for index, case in enumerate(parameters):
        case_start = time.perf_counter()
        try:
            s, i, trajectory = simulate(config, case)
        except NonlinearSolveError as error:
            raise RuntimeError(
                f"Trajectory {index} failed to solve ({case.as_dict()}): {error}"
            ) from error
        report = acceptance_report(
            trajectory,
            config.solver.positivity_tolerance,
            config.solver.balance_tolerance,
        )
        if not report["passed"]:
            raise RuntimeError(
                f"Trajectory {index} failed acceptance ({case.as_dict()}): "
                + "; ".join(report["reasons"])
            )
        susceptible[index] = s
        infectious[index] = i
        diagnostics.append(
            {"sim_id": index, **report, "wall_seconds": time.perf_counter() - case_start}
        )
        if verbose and ((index + 1) % 50 == 0 or index + 1 == n):
            print(f"  solved {index + 1}/{n} trajectories")

    splits = split_simulation_ids(n, config.splits.fractions, config.seeds.split)
    normalization = compute_normalization(susceptible, infectious, splits["train"])
    x_centers = Grid(config.domain.length, cells).centers
    coefficients = np.asarray([case.susceptible_coefficients for case in parameters])
    np.savez_compressed(
        destination / DATASET_FILE,
        x_centers=x_centers,
        times=times,
        susceptible=susceptible,
        infectious=infectious,
        diffusivity=np.asarray([case.diffusivity for case in parameters]),
        transmission=np.asarray([case.transmission for case in parameters]),
        infection_amplitude=np.asarray([case.infection_amplitude for case in parameters]),
        infection_center=np.asarray([case.infection_center for case in parameters]),
        infection_width=np.asarray([case.infection_width for case in parameters]),
        susceptible_constant=np.asarray(
            [case.susceptible_family == "constant" for case in parameters]
        ),
        susceptible_amplitude=np.asarray([case.susceptible_amplitude for case in parameters]),
        susceptible_coefficients=coefficients,
        sim_ids=np.arange(n, dtype=np.int64),
        train_ids=splits["train"],
        val_ids=splits["val"],
        test_ids=splits["test"],
    )
    split_of = {int(sim): name for name, ids in splits.items() for sim in ids}
    keys = (
        "newton_iterations_max",
        "relative_balance_residual",
        "minimum_susceptible",
        "minimum_infectious",
        "wall_seconds",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_provenance": capture_git_state(),
        "config": config.as_dict(),
        "config_sha256": config.sha256,
        "seeds": config.as_dict()["seeds"],
        "dataset_file": DATASET_FILE,
        "dataset_sha256": file_sha256(destination / DATASET_FILE),
        "storage": {
            "fields": "float32 cell averages on the learning grid",
            "shapes": {
                "susceptible": list(susceptible.shape),
                "infectious": list(infectious.shape),
            },
            "parameters": "float64",
        },
        "grid": {
            "length": config.domain.length,
            "learning_cells": cells,
            "solver_cells": config.solver_cells,
            "refinement_factor": config.solver.refinement_factor,
            "solver_dt": config.solver.dt,
            "x_centers": x_centers.tolist(),
        },
        "times": times.tolist(),
        "splits": {name: ids.tolist() for name, ids in splits.items()},
        "split_counts": {name: int(len(ids)) for name, ids in splits.items()},
        "normalization": normalization,
        "normalization_definition": (
            "per-channel population mean/std (ddof=0) over every training "
            "trajectory, snapshot, and learning-grid cell; not weighted by pair frequency"
        ),
        "solver_diagnostics": {
            "newton_iterations_max": max(d["newton_iterations_max"] for d in diagnostics),
            "backtracks_total": sum(d["backtracks_total"] for d in diagnostics),
            "maximum_relative_balance_residual": max(
                d["relative_balance_residual"] for d in diagnostics
            ),
            "minimum_susceptible": min(d["minimum_susceptible"] for d in diagnostics),
            "minimum_infectious": min(d["minimum_infectious"] for d in diagnostics),
            "solve_seconds_total": sum(d["wall_seconds"] for d in diagnostics),
        },
        "refinement": refinement,
        "trajectories": [
            {
                "sim_id": index,
                "split": split_of[index],
                "parameters": case.as_dict(),
                "diagnostics": {key: diagnostics[index][key] for key in keys},
            }
            for index, case in enumerate(parameters)
        ],
        "wall_seconds": time.perf_counter() - start,
    }
    (destination / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if verbose:
        print(
            f"Wrote {n} trajectories ({manifest['split_counts']}) to {destination} "
            f"in {manifest['wall_seconds']:.1f} s"
        )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    generate_dataset(load_config(args.config), args.output, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
