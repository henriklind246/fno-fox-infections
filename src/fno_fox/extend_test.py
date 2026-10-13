"""Append fresh held-out trajectories while preserving a checkpoint's training data.

    python -m fno_fox.extend_test --data data/fox_rabies \
        --output data/fox_rabies_test100 --count 100 --seed 6
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from fno_fox.data import (
    DATASET_FILE, MANIFEST_FILE, PARAMETER_KEYS, DatasetBundle,
    capture_git_state, file_sha256, load_dataset,
)
from fno_fox.physics import acceptance_report, simulate
from fno_fox.sampling import sample_parameters


def validate_test_extension(bundle: DatasetBundle) -> DatasetBundle:
    """Verify the complete original dataset and that every added row is test-only."""
    marker = bundle.manifest["test_extension"]
    source_root = Path(marker["source_data"]).resolve()
    if source_root == bundle.root.resolve():
        raise ValueError("Test extension cannot reference itself.")
    source = load_dataset(source_root)
    if "test_extension" in source.manifest:
        raise ValueError("Test extension must reference the original training dataset.")
    for key in ("dataset_sha256", "config_sha256"):
        if marker.get(key) != source.manifest[key]:
            raise ValueError(f"Test extension source {key} provenance does not match.")
    n = source.n_trajectories
    added = np.arange(n, bundle.n_trajectories, dtype=np.int64)
    if not len(added) or not np.array_equal(added, marker["added_test_ids"]):
        raise ValueError("Test extension must contain new consecutive test IDs.")
    for name in ("x_centers", "times"):
        if not np.array_equal(getattr(bundle, name), getattr(source, name)):
            raise ValueError(f"Test extension changed original {name}.")
    for name in ("susceptible", "infectious"):
        if not np.array_equal(getattr(bundle, name)[:n], getattr(source, name)):
            raise ValueError(f"Test extension changed original {name} fields.")
    for name in PARAMETER_KEYS:
        if not np.array_equal(bundle.parameters[name][:n], source.parameters[name]):
            raise ValueError(f"Test extension changed original {name} parameters.")
    signatures = np.column_stack([bundle.parameters[name] for name in PARAMETER_KEYS])
    original = {tuple(row) for row in signatures[:n]}
    fresh = [tuple(row) for row in signatures[n:]]
    if len(set(fresh)) != len(fresh) or any(row in original for row in fresh):
        raise ValueError("Test extension repeats an original or new trajectory specification.")
    for name in ("train", "val", "test"):
        expected = source.splits[name]
        if name == "test":
            expected = np.sort(np.concatenate((expected, added)))
        if not np.array_equal(bundle.splits[name], expected):
            raise ValueError(f"Test extension changed the original {name} split.")
    expected = extension_config(source, len(added))
    for section in ("domain", "solver", "sampling", "splits"):
        if getattr(bundle.config, section) != getattr(expected, section):
            raise ValueError(f"Test extension changed the {section} data definition.")
    if bundle.normalization != source.normalization:
        raise ValueError("Test extension changed training normalization.")
    return source


def extension_config(source: DatasetBundle, count: int):
    n = source.n_trajectories + count
    sizes = [len(source.splits[name]) + (count if name == "test" else 0)
             for name in ("train", "val", "test")]
    return replace(source.config,
                   sampling=replace(source.config.sampling, n_trajectories=n),
                   splits=replace(source.config.splits, fractions=tuple(size / n for size in sizes)))


def extend_test_dataset(data_dir, output_dir, *, count=100, seed=6, verbose=True) -> Path:
    if count < 3 or seed < 0:
        raise ValueError("Test generation requires count >= 3 and a nonnegative seed.")
    source = load_dataset(data_dir)
    if "test_extension" in source.manifest:
        raise ValueError("Extend the original dataset, not an existing extension.")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Test extension output already exists: {destination}")
    destination.mkdir(parents=True)
    start = perf_counter()
    sampling = replace(source.config.sampling, n_trajectories=count)
    parameters = sample_parameters(sampling, np.random.default_rng(seed))
    shape = (count, source.n_snapshots, len(source.x_centers))
    susceptible, infectious = np.empty(shape, np.float32), np.empty(shape, np.float32)
    diagnostics = []
    for index, case in enumerate(parameters):
        begin = perf_counter()
        s, i, trajectory = simulate(source.config, case)
        report = acceptance_report(trajectory, source.config.solver.positivity_tolerance,
                                   source.config.solver.balance_tolerance)
        if not report["passed"]:
            raise RuntimeError(f"New test trajectory {index} failed acceptance: {report['reasons']}")
        susceptible[index], infectious[index] = s, i
        diagnostics.append({**report, "wall_seconds": perf_counter() - begin})
        if verbose and ((index + 1) % 10 == 0 or index + 1 == count):
            print(f"Solved {index + 1}/{count} fresh test trajectories", flush=True)
    with np.load(source.root / DATASET_FILE) as archive:
        arrays = {key: archive[key] for key in archive.files}
    additions = {
        "susceptible": susceptible, "infectious": infectious,
        "diffusivity": np.array([p.diffusivity for p in parameters]),
        "transmission": np.array([p.transmission for p in parameters]),
        "infection_amplitude": np.array([p.infection_amplitude for p in parameters]),
        "infection_center": np.array([p.infection_center for p in parameters]),
        "infection_width": np.array([p.infection_width for p in parameters]),
        "susceptible_constant": np.array([p.susceptible_family == "constant" for p in parameters]),
        "susceptible_amplitude": np.array([p.susceptible_amplitude for p in parameters]),
        "susceptible_coefficients": np.array([p.susceptible_coefficients for p in parameters]),
    }
    for key, values in additions.items():
        arrays[key] = np.concatenate((arrays[key], values))
    added = np.arange(source.n_trajectories, source.n_trajectories + count, dtype=np.int64)
    arrays["sim_ids"] = np.arange(source.n_trajectories + count, dtype=np.int64)
    arrays["test_ids"] = np.sort(np.concatenate((arrays["test_ids"], added)))
    np.savez_compressed(destination / DATASET_FILE, **arrays)
    config = extension_config(source, count)
    manifest = deepcopy(source.manifest)
    manifest.update(created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    code_provenance=capture_git_state(), config=config.as_dict(),
                    config_sha256=config.sha256, dataset_sha256=file_sha256(destination / DATASET_FILE),
                    wall_seconds=perf_counter() - start)
    manifest["storage"]["shapes"] = {name: list(arrays[name].shape)
                                        for name in ("susceptible", "infectious")}
    manifest["splits"] = {name: arrays[f"{name}_ids"].tolist() for name in ("train", "val", "test")}
    manifest["split_counts"] = {name: len(ids) for name, ids in manifest["splits"].items()}
    manifest["test_extension"] = {
        "source_data": str(source.root.resolve()), "dataset_sha256": source.manifest["dataset_sha256"],
        "config_sha256": source.manifest["config_sha256"], "added_test_ids": added.tolist(),
        "sampling_seed": seed, "sampling_count": count,
        "split_strategy": "Preserve all original rows/splits; append only fresh test trajectories.",
        "refinement_note": "Reuse the original refinement study; physics, grid and solver are identical.",
    }
    diagnostic_keys = tuple(source.manifest["trajectories"][0]["diagnostics"])
    for sim, case, report in zip(added, parameters, diagnostics):
        manifest["trajectories"].append({"sim_id": int(sim), "split": "test",
                                         "parameters": case.as_dict(),
                                         "diagnostics": {key: report[key] for key in diagnostic_keys}})
    combined = [case["diagnostics"] for case in manifest["trajectories"]]
    manifest["solver_diagnostics"].update(
        newton_iterations_max=max(d["newton_iterations_max"] for d in combined),
        backtracks_total=source.manifest["solver_diagnostics"]["backtracks_total"]
                         + sum(d["backtracks_total"] for d in diagnostics),
        maximum_relative_balance_residual=max(d["relative_balance_residual"] for d in combined),
        minimum_susceptible=min(d["minimum_susceptible"] for d in combined),
        minimum_infectious=min(d["minimum_infectious"] for d in combined),
        solve_seconds_total=sum(d["wall_seconds"] for d in combined),
    )
    (destination / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (destination / "config.yaml").write_text(yaml.safe_dump(config.as_dict(), sort_keys=False), encoding="utf-8")
    validate_test_extension(load_dataset(destination))
    if verbose:
        print(f"Verified {count} new test cases; preserved all original data. Output: {destination}", flush=True)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=6)
    args = parser.parse_args(argv)
    extend_test_dataset(args.data, args.output, count=args.count, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
