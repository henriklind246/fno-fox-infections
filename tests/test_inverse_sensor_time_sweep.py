from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from fno_fox.config import InverseConfig, config_from_dict
from fno_fox.data import FieldEncoder, load_dataset
from fno_fox.generate import generate_dataset
from fno_fox.inverse import METRIC_SCOPE_LABELS, evaluate_case, field_metrics, validate_inputs
from fno_fox.inverse_numerics import FieldObjective, joint_grid, recover, shared_joint_grids
from fno_fox.inverse_sensor_sweep import select_sweep_cases, sensor_sweep
from fno_fox.inverse_sensor_time_sweep import (
    main, nested_time_layouts, observation_layouts, sensor_time_sweep,
)
from fno_fox.train import load_checkpoint, train
from test_inverse import linear_objective
from test_inverse_sensor_sweep import fake_bundle


@pytest.fixture(scope="module", autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def time_fixture(tmp_path_factory, tiny_config):
    config = config_from_dict(tiny_config.as_dict(), {
        "domain": {"n_snapshots": 7}, "training": {"epochs": 1},
        "inverse": {"sweep_cases": 1, "sweep_time_counts": [6, 5, 4, 3, 2], "grid_size": 3,
                    "batch_size": 8, "max_iterations": 2, "max_evaluations": 5, "edge_max_evaluations": 3}})
    root = tmp_path_factory.mktemp("sensor_time_fixture")
    generate_dataset(config, root / "data", verbose=False)
    train(config, root / "data", root / "run", device="cpu", verbose=False)
    return config, root / "data", root / "run" / "best.pt"


def test_time_defaults_and_older_checkpoint_configs(default_config):
    assert default_config.inverse.sweep_time_counts == (20, 15, 10, 7, 5)
    payload = default_config.as_dict()
    payload["inverse"].pop("sweep_time_counts")
    assert config_from_dict(payload).inverse.sweep_time_counts == (20, 15, 10, 7, 5)
    assert config_from_dict(payload).inverse.sweep_cases == 100
    assert config_from_dict(payload).inverse.pilot_cases == 10


@pytest.mark.parametrize("counts", [[], [20, 20], [1, 5], [5.5, 10], [5.0, 10], [True, 5], None, "20,15"])
def test_invalid_time_counts(counts, default_config):
    with pytest.raises(ValueError, match="sweep_time_counts"):
        config_from_dict(default_config.as_dict(), {"inverse": {"sweep_time_counts": counts}})


def test_fifteen_uniform_spatial_and_nested_temporal_layouts():
    times = np.arange(1, 21) / 4
    x = np.linspace(.01, 9.99, 256)
    layouts = observation_layouts(x, times, (20, 15, 10, 7, 5))
    assert len(layouts) == 15
    temporal = nested_time_layouts(times, (20, 15, 10, 7, 5))
    assert np.array_equal(temporal[5], np.rint(np.linspace(0, 19, 5)).astype(int))
    assert np.array_equal(temporal[20], np.arange(20))
    previous = set()
    for count in sorted(temporal):
        assert previous <= set(temporal[count])
        previous = set(temporal[count])
        assert len(previous) == count
        assert temporal[count][0] == 0 and temporal[count][-1] == 19
        assert np.all(np.diff(temporal[count]) > 0)
    again = observation_layouts(x, times, (5, 20, 10, 15, 7))
    for key, layout in layouts.items():
        n, t = layout["sensor_count"], layout["time_count"]
        assert np.array_equal(layout["indices"], np.rint(np.linspace(0, 255, n)).astype(int))
        assert np.array_equal(layout["time_indices"], temporal[t])
        assert np.array_equal(layout["times"], times[temporal[t]])
        assert np.array_equal(layout["time_indices"], again[key]["time_indices"])
        assert layout["observed_scalars"] == 2 * n * t


def test_temporal_ties_choose_earlier_even_with_float_roundoff():
    result = nested_time_layouts(np.arange(1, 9), (2, 3, 4, 5))
    assert np.array_equal(result[3], [0, 3, 7])  # midpoint 4.5 -> timestamp 4
    assert np.array_equal(result[4], [0, 3, 5, 7])
    assert np.array_equal(result[5], [0, 1, 3, 5, 7])
    result = nested_time_layouts(np.arange(1, 11) / 10, (2, 3))
    assert np.array_equal(result[3], [0, 4, 9])
    result = nested_time_layouts([1, 2, 3, 7, 8, 10], (2, 3, 4))
    assert np.array_equal(result[3], [0, 3, 5])
    assert np.array_equal(result[4], [0, 2, 3, 5])


@pytest.mark.parametrize("times", [[0, 1, 2], [1, 1, 2], [3, 2, 1], [1, np.nan, 3], []])
def test_bad_positive_times(times):
    with pytest.raises(ValueError, match="positive"):
        nested_time_layouts(times, (2,))


def test_identical_hundred_case_selection_across_fifteen_configs(tmp_path, default_config, monkeypatch):
    import fno_fox.inverse_sensor_time_sweep as module
    bundle = fake_bundle()
    bundle.times = np.arange(21) / 4
    bundle.x_centers = np.arange(256)
    monkeypatch.setattr(module, "load_dataset", lambda p: bundle)
    captured = {}
    def runner(config, actual_bundle, checkpoint, output, ids, layouts, **kwargs):
        captured.update(ids=ids, layouts=layouts, **kwargs)
        return {"failed_cases": []}
    monkeypatch.setattr(module, "_run_sweep", runner)
    sensor_time_sweep(default_config, "unused", "unused", tmp_path / "output", figures=False)
    ids = captured["ids"]
    assert len(ids) == len(np.unique(ids)) == 100
    assert bundle.parameters["susceptible_constant"][ids].sum() == 50
    assert np.array_equal(ids, select_sweep_cases(bundle, default_config.inverse))
    assert len(captured["layouts"]) == 15
    assert captured["pipeline"] == "fno_fox.inverse_sensor_time_sweep"


@pytest.mark.parametrize("problem", ["cases", "cells", "times"])
def test_inadequate_inputs_fail_before_checkpoint_and_output(tmp_path, default_config, monkeypatch, problem):
    import fno_fox.inverse_sensor_time_sweep as module
    bundle = fake_bundle(90 if problem == "cases" else 110)
    bundle.x_centers = np.arange(15 if problem == "cells" else 16)
    bundle.times = np.arange(20 if problem == "times" else 21) / 4
    monkeypatch.setattr(module, "load_dataset", lambda p: bundle)
    def should_not_run(*args, **kwargs):
        pytest.fail("Runner/model reached before layout or cardinality check")
    monkeypatch.setattr(module, "_run_sweep", should_not_run)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match={"cases": "100.*90", "cells": "16", "times": "20.*19"}[problem]):
        sensor_time_sweep(default_config, "unused", "unused.pt", output)
    assert not output.exists()


@pytest.mark.parametrize("indices", [[0, 0], [-1], [3], [0.5], [], [2, 0], np.array([2, 0], dtype=np.uint64), [False, True]])
def test_invalid_objective_time_indices(indices):
    full = linear_objective()
    with pytest.raises(ValueError, match="Time indices"):
        FieldObjective(full.model, full.encoder, full.obs, full.device, 4, time_indices=indices)


def test_excluded_space_and_time_targets_do_not_change_recovery():
    full = linear_objective()
    sensors, times = np.array([0, 7]), np.array([0, 2])
    sparse = FieldObjective(full.model, full.encoder, full.obs, full.device, 4, sensors, time_indices=times)
    normalized, physical = full.obs.targets.copy(), full.obs.physical_targets.copy()
    normalized[1] += 20
    physical[1] += 20
    normalized[:, 1:7] += 30
    physical[:, 1:7] += 30
    changed = replace(full.obs, targets=normalized, physical_targets=physical)
    other = FieldObjective(full.model, full.encoder, changed, full.device, 4, sensors, time_indices=times)
    points = np.array([[0, 0], [.31, .67], [1, 1]])
    assert np.array_equal(sparse.losses(points), other.losses(points))
    config = InverseConfig(max_iterations=5, max_evaluations=12, edge_max_evaluations=5)
    a, b = [recover(obj, joint_grid(obj.losses, 3), config, 0) for obj in (sparse, other)]
    assert np.array_equal(a["arms"]["8"]["z"], b["arms"]["8"]["z"])
    pred = full.physical_prediction([.31, .67])
    assert field_metrics(sparse.observe(pred), sparse.observe(physical), sparse.observed_times)["S"]["rmse"] == 0
    assert field_metrics(pred[times], physical[times], full.obs.times[times])["S"]["rmse"] > 1
    assert field_metrics(pred, physical, full.obs.times)["S"]["rmse"] > 1


@pytest.mark.parametrize("point", [[.4, .6], [0., 1.]])
def test_space_time_jacobian_slices_scaling_and_finite_differences(point):
    full = linear_objective(batch=2)
    sensors, times = np.array([0, 3, 7]), np.array([0, 2])
    sparse = FieldObjective(full.model, full.encoder, full.obs, full.device, 2, sensors, time_indices=times)
    physical, scaled = sparse.jacobians(point)
    a, b = full.jacobians(point)
    assert physical.shape == (2, 3, 2, 2)
    assert np.array_equal(physical, a[times][:, sensors])
    expected = b.reshape(3, 8, 2, 2)[times][:, sensors].reshape(-1, 2) * np.sqrt(24 / 6)
    assert np.array_equal(scaled, expected)
    assert len(scaled) == 12
    std = np.array([full.encoder.s_std, full.encoder.i_std])
    assert np.allclose(scaled.reshape(physical.shape), physical * full.ranges / std[None, None, :, None] / np.sqrt(12))
    for h in (1e-2, 3e-3, 1e-3):
        derivatives = []
        for k in range(2):
            lo, hi = np.array(point), np.array(point)
            lo[k], hi[k] = max(0, lo[k] - h), min(1, hi[k] + h)
            values = sparse.observe(full.predict_many(np.array([lo, hi])).astype(float))
            derivatives.append(((values[1] - values[0]) / (hi[k] - lo[k])).ravel())
        assert np.allclose(np.column_stack(derivatives) / np.sqrt(12), scaled, rtol=2e-4, atol=1e-5)


def test_full_time_selection_preserves_loss_grid_recovery_and_jacobian():
    full = linear_objective()
    # Twenty saved positive times, with exactly noiseless targets from the initial fields.
    times = np.linspace(.25, 5, 20)
    obs = replace(full.obs, times=times, targets=np.zeros((20, 8, 2), np.float32), physical_targets=np.zeros((20, 8, 2)))
    full = FieldObjective(full.model, full.encoder, obs, full.device, 4, [0, 3, 7])
    values = full.predict_many([[.31, .67]])[0]
    obs = replace(obs, targets=values, physical_targets=np.stack(full.encoder.decode(values), axis=-1))
    old = FieldObjective(full.model, full.encoder, obs, full.device, 4, [0, 3, 7])
    new = FieldObjective(full.model, full.encoder, obs, full.device, 4, [0, 3, 7], time_indices=np.arange(20))
    grids = [joint_grid(obj.losses, 3) for obj in (old, new)]
    assert np.array_equal(grids[0]["loss"], grids[1]["loss"])
    for a, b in zip(old.jacobians([.4, .6]), new.jacobians([.4, .6])):
        assert np.array_equal(a, b)
    config = InverseConfig(max_iterations=5, max_evaluations=12, edge_max_evaluations=5)
    recovered = [recover(obj, grid, config, 0) for obj, grid in zip((old, new), grids)]
    for arm in ("4", "8"):
        assert np.array_equal(recovered[0]["arms"][arm]["z"], recovered[1]["arms"][arm]["z"])


def test_shared_grid_and_optimizer_cache_use_full_time_predictions_once():
    full = linear_objective(batch=2)
    layouts = {"all": {"indices": np.arange(8), "time_indices": np.arange(3)},
               "selected": {"indices": np.array([0, 7]), "time_indices": np.array([0, 2])}}
    full.model.query_sizes.clear()
    grids, cost = shared_joint_grids(full, layouts, 5)
    assert sum(full.model.query_sizes) == 25 * 3
    assert cost["field_queries"] == 75
    cache, stats = OrderedDict(), {"parameter_predictions": 0, "field_queries": 0, "cache_hits": 0}
    for key, layout in layouts.items():
        obj = FieldObjective(full.model, full.encoder, full.obs, full.device, 2, layout["indices"],
                             cache, stats, time_indices=layout["time_indices"])
        assert np.array_equal(grids[key]["loss"], joint_grid(obj.losses, 5)["loss"])
        obj([.31, .67])
    assert stats["cache_hits"] >= 1
    assert cache[(.31, .67)].shape == (1, 3, 8, 2)


def assert_metrics(actual, expected):
    for channel in ("S", "I"):
        for metric in ("bias", "rmse", "relative_l2"):
            assert actual[channel][metric] == pytest.approx(expected[channel][metric], rel=1e-12, abs=1e-15)
        for a, b in zip(actual[channel]["by_time"], expected[channel]["by_time"]):
            assert a == pytest.approx(b, rel=1e-12, abs=1e-15)
        assert len(actual[channel]["by_time"]) == len(expected[channel]["by_time"])


def test_fifteen_configuration_fixture_artifacts_metrics_pairing_and_equivalence(tmp_path, time_fixture):
    config, data, checkpoint = time_fixture
    root = tmp_path / "sweep"
    report = sensor_time_sweep(config, data, checkpoint, root, device="cpu", verbose=False)
    assert report["failed_cases"] == [] and report["completed_configuration_cases"] == 15
    manifest = json.loads((root / "manifest.json").read_text())
    sim = manifest["selected_ids"][0]
    assert manifest["status"] == "completed" and len(manifest["observation_layouts"]) == 15
    starts = None
    all_rows = {}
    with np.load(root / "validation_residuals.npz") as shared:
        full_residuals = shared["residuals"]
    for key, layout in manifest["observation_layouts"].items():
        directory = root / "configurations" / key / "cases" / str(sim)
        for name in ("grid.npz", "optimizer.json", "optimizer_trace.csv", "jacobians.npz", "fields.npz", "metrics.json",
                     "sensitivities.json", "result.json", "loss_surface.png", "reconstruction.png"):
            assert (directory / name).stat().st_size > 0
        rows = json.loads((directory / "result.json").read_text())["rows"]
        all_rows[key] = rows
        assert {r["sim_id"] for r in rows} == {sim}
        assert {r["observed_scalars"] for r in rows} == {2 * layout["sensor_count"] * layout["time_count"]}
        optimizer = json.loads((directory / "optimizer.json").read_text())
        current_starts = [r["start"] for r in optimizer["starts"]][1:]
        if starts is None:
            starts = current_starts
        assert starts == current_starts
        assert optimizer["seed"] == config.inverse.start_seed + sim
        with np.load(directory / "jacobians.npz") as jac:
            assert jac["truth_physical"].shape == (layout["time_count"], layout["sensor_count"], 2, 2)
            assert len(jac["truth_scaled"]) == layout["observed_scalars"]
            assert np.array_equal(jac["times"], layout["times"])
        metrics = json.loads((directory / "metrics.json").read_text())
        assert metrics["scope_labels"] == METRIC_SCOPE_LABELS
        with np.load(directory / "fields.npz") as fields:
            ti, si = fields["time_indices"], fields["sensor_indices"]
            assert fields["observed"].shape == (6, 64, 2)
            for arm in ("4", "8"):
                fno, fv, truth = fields[f"fno_{arm}"], fields[f"fv_{arm}"], fields["observed"]
                for scope, time_idx, sensor_idx in (("sensor", ti, si), ("full_grid_selected_times", ti, np.arange(64)),
                                                    ("full_grid", np.arange(6), np.arange(64))):
                    for comparison, a, b in (("fno_vs_observation", fno, truth), ("fv_vs_observation", fv, truth), ("fno_vs_fv", fno, fv)):
                        expected = field_metrics(a[time_idx][:, sensor_idx], b[time_idx][:, sensor_idx], fields["times"][time_idx])
                        assert_metrics(metrics[arm][scope][comparison], expected)
        with np.load(root / "configurations" / key / "validation_residuals.npz") as residuals:
            assert np.array_equal(residuals["residuals"], full_residuals[:, layout["time_indices"]][:, :, layout["indices"]])
    for name in ("cases.csv", "summary.json", "paired_changes.csv", "shared_costs.json", "sensor_time_recovery.png",
                 "sensor_time_replay_errors_sensor.png", "sensor_time_replay_errors_full_grid_selected_times.png",
                 "sensor_time_replay_errors_full_grid.png", "sensor_time_sensitivities.png", "sensor_time_start_stability.png"):
        assert (root / name).stat().st_size > 0
    assert len(report["paired_changes"]["temporal"]) == 12
    assert len(report["paired_changes"]["spatial"]) == 10
    for axis, comparisons in report["paired_changes"].items():
        for key, arms in comparisons.items():
            comparison = arms["8"]
            reference = comparison["reference_configuration"]
            assert comparison["paired_count"] == 1 and comparison["unpaired_count"] == 0
            if axis == "temporal":
                assert reference.endswith("times_6")
            else:
                assert reference.startswith("sensors_16_")
            expected = all_rows[key][1]["D_abs_error"] - all_rows[reference][1]["D_abs_error"]
            assert comparison["changes"]["D_abs_error_change"]["median"] == expected
    original = tmp_path / "spatial"
    sensor_sweep(config, data, checkpoint, original, sim_ids=[sim], device="cpu", figures=False, verbose=False)
    for n, old_name in ((16, "sparse_b"), (8, "sparse_c"), (4, "sparse_d")):
        old = json.loads((original / "configurations" / old_name / "cases" / str(sim) / "result.json").read_text())["rows"]
        for a, b in zip(old, all_rows[f"sensors_{n}_times_6"]):
            for metric in ("D_hat", "beta_hat", "loss", "singular_value_1", "condition_number", "column_cosine",
                           "fv_vs_observation_sensor_S_rmse", "fv_vs_observation_full_grid_I_rmse"):
                assert a[metric] == pytest.approx(b[metric], abs=1e-15)
    costs = json.loads((root / "shared_costs.json").read_text())["cases"][str(sim)]
    assert costs["grid"]["parameter_predictions"] == 9
    assert costs["grid"]["field_queries"] == 54
    assert costs["diagnostics"]["unique_jacobian_points"] < 15 * 4
    assert costs["wall_seconds_including_grid"] > 0
    for path in root.rglob("*.json"):
        assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()


def test_time_failures_paired_exclusions_overwrite_and_nonzero_exit(tmp_path, time_fixture, monkeypatch):
    import fno_fox.inverse_sensor_sweep as engine
    import fno_fox.inverse_sensor_time_sweep as module
    config, data, checkpoint = time_fixture
    config = replace(config, inverse=replace(config.inverse, sweep_cases=2))
    original = engine.evaluate_case
    def sometimes_fail(*args, **kwargs):
        if kwargs["configuration"] == "sensors_16_times_6":
            raise RuntimeError("Fixture failed temporal reference")
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, "evaluate_case", sometimes_fail)
    root = tmp_path / "failures"
    report = sensor_time_sweep(config, data, checkpoint, root, figures=False, verbose=False)
    assert len(report["failed_cases"]) == 2
    assert report["paired_changes"]["temporal"]["sensors_16_times_5"]["8"]["paired_count"] == 0
    assert report["paired_changes"]["spatial"]["sensors_8_times_6"]["8"]["unpaired_count"] == 2
    assert report["paired_changes"]["spatial"]["sensors_8_times_5"]["8"]["paired_count"] == 2
    for sim in report["selected_ids"]:
        directory = root / "configurations" / "sensors_16_times_6" / "cases" / str(sim)
        assert (directory / "failure.json").exists() and not (directory / "result.json").exists()
    with pytest.raises(FileExistsError):
        sensor_time_sweep(config, data, checkpoint, root, figures=False)
    stale = root / "sensor_time_replay_errors_full_grid.png"
    stale.write_bytes(b"stale")
    monkeypatch.setattr(module, "load_config", lambda p: config)
    assert main(["--config", "unused", "--data", str(data), "--checkpoint", str(checkpoint),
                 "--output", str(root), "--overwrite"]) == 1
    assert stale.read_bytes() != b"stale"
    assert json.loads((root / "manifest.json").read_text())["status"] == "completed_with_failures"


def test_time_masks_share_full_jacobians_diagnostic_predictions_and_fv(tmp_path, time_fixture, monkeypatch):
    import fno_fox.inverse as module
    config, data, checkpoint = time_fixture
    bundle = load_dataset(data)
    model, payload = load_checkpoint(checkpoint)
    encoder = FieldEncoder.from_dict(payload["encoder"])
    # Checkpoint configurations predating temporal sweeps remain valid.
    payload["config"]["inverse"].pop("sweep_time_counts")
    assert validate_inputs(config, bundle, payload) == encoder
    sim = int(bundle.splits["test"][0])
    calls = {"jacobian": 0, "prediction": 0, "fv": 0}
    jacobian, prediction, fv = FieldObjective.full_jacobians, FieldObjective.physical_prediction, module.fv_replay
    def count_jacobian(self, *args):
        calls["jacobian"] += 1
        return jacobian(self, *args)
    def count_prediction(self, *args):
        calls["prediction"] += 1
        return prediction(self, *args)
    def count_fv(*args):
        calls["fv"] += 1
        return fv(*args)
    monkeypatch.setattr(FieldObjective, "full_jacobians", count_jacobian)
    monkeypatch.setattr(FieldObjective, "physical_prediction", count_prediction)
    monkeypatch.setattr(module, "fv_replay", count_fv)
    shared = {}
    for iteration, times in enumerate((np.arange(6), np.arange(6), np.array([0, 5]))):
        evaluate_case(bundle, sim, model, encoder, torch.device("cpu"), config.inverse, tmp_path / str(iteration),
                      sensor_indices=np.array([0, 63]), time_indices=times, shared=shared)
        if iteration == 0:
            first = calls.copy()
        elif iteration == 1:
            assert calls == first
    assert calls["jacobian"] == len(shared["jacobians"])
    assert calls["prediction"] == len(shared["predictions"])
    assert calls["fv"] == len(shared["fv"])


def test_selection_manifest_precedes_inference_and_global_failure_is_recorded(tmp_path, time_fixture, monkeypatch):
    import fno_fox.inverse_sensor_sweep as engine
    config, data, checkpoint = time_fixture
    root = tmp_path / "manifest_first"
    def before_inference(*args):
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["status"] == "running" and len(manifest["selected_ids"]) == 1
        assert len(manifest["observation_layouts"]) == 15
        assert manifest["metric_scopes"] == METRIC_SCOPE_LABELS
        raise RuntimeError("Fixture global validation failure")
    monkeypatch.setattr(engine, "validation_residuals", before_inference)
    with pytest.raises(RuntimeError, match="global validation"):
        sensor_time_sweep(config, data, checkpoint, root, figures=False, verbose=False)
    assert json.loads((root / "manifest.json").read_text())["status"] == "failed"


def test_shared_grid_failure_marks_all_fifteen_configuration_cases(tmp_path, time_fixture, monkeypatch):
    import fno_fox.inverse_sensor_sweep as engine
    config, data, checkpoint = time_fixture
    def fail(*args):
        raise ValueError("Fixture shared grid failure")
    monkeypatch.setattr(engine, "shared_joint_grids", fail)
    root = tmp_path / "grid_failure"
    report = sensor_time_sweep(config, data, checkpoint, root, figures=False, verbose=False)
    assert report["failed_configuration_cases"] == 15
    assert report["completed_configuration_cases"] == 0
    for key in report["configurations"]:
        sim = report["selected_ids"][0]
        assert (root / "configurations" / key / "cases" / str(sim) / "failure.json").exists()


def test_cli_options_and_output_guards(tmp_path, time_fixture, monkeypatch):
    import fno_fox.inverse_sensor_time_sweep as module
    config, data, checkpoint = time_fixture
    with pytest.raises(ValueError, match="input"):
        sensor_time_sweep(config, data, checkpoint, data, overwrite=True, figures=False)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    with pytest.raises(ValueError, match="existing fno_fox.inverse_sensor_time_sweep"):
        sensor_time_sweep(config, data, checkpoint, unrelated, overwrite=True, figures=False)
    wrong_pipeline = tmp_path / "spatial"
    wrong_pipeline.mkdir()
    (wrong_pipeline / "manifest.json").write_text(json.dumps({"pipeline": "fno_fox.inverse_sensor_sweep"}))
    with pytest.raises(ValueError, match="inverse_sensor_time_sweep"):
        sensor_time_sweep(config, data, checkpoint, wrong_pipeline, overwrite=True, figures=False)
    monkeypatch.setattr(module, "load_config", lambda p: config)
    captured = {}
    def run(*args, **kwargs):
        captured.update(kwargs)
        return {"failed_cases": []}
    monkeypatch.setattr(module, "sensor_time_sweep", run)
    assert main(["--config", "c", "--data", "d", "--checkpoint", "p", "--output", "o",
                 "--device", "cpu", "--sim-ids", "1", "--overwrite"]) == 0
    assert captured["device"] == "cpu" and captured["sim_ids"] == [1] and captured["overwrite"]
    assert main(["--config", "c", "--data", "d", "--checkpoint", "p", "--output", "o", "--all-test"]) == 0
    assert captured["all_test"]
    assert main(["--config", "c", "--data", "d", "--checkpoint", "p", "--output", "o",
                 "--resume", "--max-recovery-problems", "500"]) == 0
    assert captured["resume"] and captured["max_recovery_problems"] == 500
    with pytest.raises(SystemExit):
        main(["--config", "c", "--data", "d", "--checkpoint", "p", "--output", "o", "--all-test", "--sim-ids", "1"])


def test_capped_resume_reuses_successes_and_rejects_changed_inputs(tmp_path, time_fixture, monkeypatch):
    import fno_fox.inverse_sensor_sweep as engine
    config, data, checkpoint = time_fixture
    config = replace(config, inverse=replace(config.inverse, sweep_cases=2))
    ids = list(map(int, load_dataset(data).splits["test"][:2]))
    original = engine.evaluate_case
    calls = []
    def interrupt(*args, **kwargs):
        if len(calls) == 3:
            raise KeyboardInterrupt()
        calls.append((args[1], kwargs["configuration"]))
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, "evaluate_case", interrupt)
    root = tmp_path / "resume"
    with pytest.raises(KeyboardInterrupt):
        sensor_time_sweep(config, data, checkpoint, root, device="cpu", sim_ids=ids,
                          figures=False, verbose=False)
    saved = {path: path.read_bytes() for path in root.rglob("result.json")}
    assert len(saved) == 3
    assert json.loads((root / "manifest.json").read_text())["status"] == "interrupted"
    before = (root / "manifest.json").read_bytes()
    changed = replace(config, inverse=replace(config.inverse, grid_size=5))
    with pytest.raises(ValueError, match="changed recovery settings"):
        sensor_time_sweep(changed, data, checkpoint, root, device="cpu", sim_ids=ids,
                          figures=False, resume=True, max_recovery_problems=20)
    assert (root / "manifest.json").read_bytes() == before
    copied_checkpoint = tmp_path / "changed.pt"
    payload = torch.load(checkpoint, weights_only=False)
    payload["epoch"] += 1
    torch.save(payload, copied_checkpoint)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        sensor_time_sweep(config, data, copied_checkpoint, root, device="cpu", sim_ids=ids,
                          figures=False, resume=True, max_recovery_problems=20)
    calls.clear()
    def record(*args, **kwargs):
        calls.append((args[1], kwargs["configuration"]))
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, "evaluate_case", record)
    report = sensor_time_sweep(config, data, checkpoint, root, device="cpu", sim_ids=ids,
                               figures=False, verbose=False, resume=True, max_recovery_problems=20)
    assert len(calls) == 17 and report["completed_configuration_cases"] == 20
    assert report["failed_cases"] == [] and report["budget_excluded_configuration_cases"] == 10
    assert len(set(calls)) == len(calls)
    for path, content in saved.items():
        assert path.read_bytes() == content
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["resume"]["reused_configuration_cases"] == 3
    for key, scheduled in manifest["scheduled_ids_by_configuration"].items():
        assert scheduled == (ids if key.startswith("sensors_16_") else ids[:1])
        assert report["configurations"][key]["completed_cases"] == len(scheduled)
    calls.clear()
    sensor_time_sweep(config, data, checkpoint, root, device="cpu", sim_ids=ids,
                      figures=False, verbose=False, resume=True, max_recovery_problems=20)
    assert calls == []


@pytest.mark.parametrize("budget", [0, -1, True, 2.5, 16])
def test_invalid_recovery_budget_before_output(tmp_path, time_fixture, budget):
    config, data, checkpoint = time_fixture
    root = tmp_path / "invalid"
    with pytest.raises(ValueError, match="max_recovery_problems"):
        sensor_time_sweep(config, data, checkpoint, root, figures=False, max_recovery_problems=budget)
    assert not root.exists()
