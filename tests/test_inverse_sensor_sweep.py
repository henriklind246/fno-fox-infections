from __future__ import annotations

from dataclasses import replace
from collections import OrderedDict
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fno_fox.config import InverseConfig, config_from_dict
from fno_fox.data import load_dataset
from fno_fox.inverse import evaluate_case, field_metrics, inverse
from fno_fox.inverse_numerics import FieldObjective, joint_grid, recover, shared_joint_grids
from fno_fox.inverse_sensor_sweep import (
    main, select_sweep_cases, sensor_layouts, sensor_sweep,
)
from fno_fox.train import load_checkpoint, train
from test_inverse import linear_objective


@pytest.fixture(scope="module", autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def sweep_checkpoint(tmp_path_factory, tiny_config, tiny_dataset):
    path = tmp_path_factory.mktemp("sensor_sweep_training") / "run"
    train(tiny_config, tiny_dataset, path, device="cpu", verbose=False)
    return path / "best.pt"


def fake_bundle(n=150, constant_count=75):
    return SimpleNamespace(splits={"test": np.arange(10, n + 10)},
                           parameters={"susceptible_constant": np.arange(n + 10) < constant_count + 10})


def test_sweep_defaults_and_backward_compatibility(default_config):
    assert default_config.inverse.sweep_cases == 100
    assert default_config.inverse.pilot_cases == 10
    payload = default_config.as_dict()
    payload["inverse"].pop("sweep_cases")
    assert config_from_dict(payload).inverse.sweep_cases == 100
    with pytest.raises(ValueError, match="inverse"):
        config_from_dict(payload, {"inverse": {"sweep_cases": 0}})


def test_exact_reproducible_balanced_selection_and_explicit_ids():
    bundle = fake_bundle()
    config = InverseConfig()
    ids = select_sweep_cases(bundle, config)
    assert len(ids) == len(np.unique(ids)) == 100
    assert np.array_equal(ids, select_sweep_cases(bundle, config))
    assert bundle.parameters["susceptible_constant"][ids].sum() == 50
    assert np.isin(ids, bundle.splits["test"]).all()
    assert np.array_equal(select_sweep_cases(bundle, config, ids[::-1]), ids)
    assert len(select_sweep_cases(bundle, config, all_test=True)) == 150
    bundle = fake_bundle(120, 8)
    ids = select_sweep_cases(bundle, config)
    assert len(ids) == 100 and bundle.parameters["susceptible_constant"][ids].sum() == 8
    with pytest.raises(ValueError, match="exactly 100"):
        select_sweep_cases(bundle, config, ids[:99])
    with pytest.raises(ValueError, match="unique"):
        select_sweep_cases(bundle, config, [10] * 100)
    with pytest.raises(ValueError, match="test split"):
        select_sweep_cases(bundle, config, np.arange(100))
    with pytest.raises(ValueError):
        select_sweep_cases(bundle, config, ids, all_test=True)


@pytest.mark.parametrize("all_test", [False, True])
def test_ninety_cases_rejected_before_loading_checkpoint_or_inference(tmp_path, tiny_config, monkeypatch, all_test):
    import fno_fox.inverse_sensor_sweep as module
    monkeypatch.setattr(module, "load_dataset", lambda p: fake_bundle(90))
    def should_not_load(*args, **kwargs):
        pytest.fail("Checkpoint loaded before cardinality check")
    monkeypatch.setattr(module, "load_checkpoint", should_not_load)
    output = tmp_path / "sweep"
    with pytest.raises(ValueError, match="at least 100.*has 90"):
        sensor_sweep(tiny_config, "unused", "unused.pt", output, all_test=all_test)
    assert not output.exists()


@pytest.mark.parametrize("nx", [32, 64, 256])
def test_uniform_unique_sensor_layouts(nx):
    x = (np.arange(nx) + .5) * 10 / nx
    layouts = sensor_layouts(x)
    assert [v["sensor_count"] for v in layouts.values()] == [nx, 32, 16, 8, 4]
    for layout in layouts.values():
        idx = layout["indices"]
        assert len(np.unique(idx)) == len(idx)
        assert idx[0] == 0 and idx[-1] == nx - 1
        assert np.array_equal(idx, np.rint(np.linspace(0, nx - 1, len(idx))).astype(int))
        assert np.array_equal(layout["x_centers"], x[idx])
        assert np.diff(idx).max() - np.diff(idx).min() <= 1
    with pytest.raises(ValueError, match="32"):
        sensor_layouts(x[:31])


def test_full_selection_matches_original_loss_jacobian_and_recovery():
    full = linear_objective()
    selected = FieldObjective(full.model, full.encoder, full.obs, full.device, full.batch_size, np.arange(8))
    points = np.array([[0., 0.], [.4, .6], [1., 1.]])
    assert np.array_equal(full.losses(points), selected.losses(points))
    for a, b in zip(full.jacobians([.4, .6]), selected.jacobians([.4, .6])):
        assert np.array_equal(a, b)
    grids, _ = shared_joint_grids(full, {"baseline": np.arange(8)}, 5)
    grid = joint_grid(full.losses, 5)
    assert np.array_equal(grids["baseline"]["loss"], grid["loss"])
    first = recover(full, grid, InverseConfig(), 3)
    second = recover(selected, grids["baseline"], InverseConfig(), 3)
    for arm in ("4", "8"):
        assert np.array_equal(first["arms"][arm]["z"], second["arms"][arm]["z"])
        assert first["arms"][arm]["loss"] == second["arms"][arm]["loss"]


def test_unobserved_targets_cannot_change_fitting_but_change_full_grid_metrics():
    full = linear_objective()
    idx = np.array([0, 3, 7])
    sparse = FieldObjective(full.model, full.encoder, full.obs, full.device, 4, idx)
    changed_targets = full.obs.targets.copy()
    changed_physical = full.obs.physical_targets.copy()
    other = np.setdiff1d(np.arange(8), idx)
    changed_targets[:, other, :] += 10
    changed_physical[:, other, :] += 10
    obs = replace(full.obs, targets=changed_targets, physical_targets=changed_physical)
    changed = FieldObjective(full.model, full.encoder, obs, full.device, 4, idx)
    points = np.array([[0., 0.], [.4, .6], [1., 1.]])
    assert np.array_equal(sparse.losses(points), changed.losses(points))
    config = InverseConfig(max_iterations=10, max_evaluations=30)
    a = recover(sparse, joint_grid(sparse.losses, 5), config, 0)
    b = recover(changed, joint_grid(changed.losses, 5), config, 0)
    assert np.array_equal(a["arms"]["8"]["z"], b["arms"]["8"]["z"])
    prediction = full.physical_prediction([.31, .67])
    assert field_metrics(prediction, obs.physical_targets, obs.times)["S"]["rmse"] > 1
    assert field_metrics(sparse.observe(prediction), sparse.observe(obs.physical_targets), obs.times)["S"]["rmse"] == 0


@pytest.mark.parametrize("point", [[.4, .6], [0., 1.]])
def test_selected_jacobians_scaling_and_finite_differences(point):
    full = linear_objective(batch=2)
    indices = np.array([0, 3, 7])
    sparse = FieldObjective(full.model, full.encoder, full.obs, full.device, 2, indices)
    physical, scaled = sparse.jacobians(point)
    full_physical, full_scaled = full.jacobians(point)
    assert physical.shape == (3, 3, 2, 2)
    assert np.array_equal(physical, full_physical[:, indices])
    expected = full_scaled.reshape(3, 8, 2, 2)[:, indices].reshape(-1, 2) * np.sqrt(8 / 3)
    assert np.allclose(scaled, expected)
    assert len(scaled) == 2 * 3 * 3
    point = np.array(point)
    for h in (1e-2, 3e-3, 1e-3):
        derivatives = []
        for k in range(2):
            lo, hi = point.copy(), point.copy()
            lo[k] = max(0, lo[k] - h)
            hi[k] = min(1, hi[k] + h)
            pair = sparse.observe(full.predict_many(np.array([lo, hi])).astype(float))
            derivatives.append(((pair[1] - pair[0]) / (hi[k] - lo[k])).ravel())
        assert np.allclose(np.column_stack(derivatives) / np.sqrt(18), scaled, rtol=2e-4, atol=1e-5)


def test_shared_grid_predictions_once_and_separate_masked_losses():
    full = linear_objective(batch=4)
    layouts = {"baseline": np.arange(8), "sparse": np.array([0, 7])}
    full.model.query_sizes.clear()
    grids, cost = shared_joint_grids(full, layouts, 7)
    assert sum(full.model.query_sizes) == 49 * 3
    assert cost["parameter_predictions"] == 49 and cost["field_queries"] == 147
    for key, indices in layouts.items():
        objective = FieldObjective(full.model, full.encoder, full.obs, full.device, 4, indices)
        expected = joint_grid(objective.losses, 7)
        assert np.array_equal(grids[key]["loss"], expected["loss"])
        assert grids[key]["evaluations"] == 0
        assert grids[key]["logical_grid_evaluations"] == 49


def test_bounded_prediction_cache_is_shared_across_sensor_objectives():
    full = linear_objective(batch=4)
    cache = OrderedDict()
    stats = {"parameter_predictions": 0, "field_queries": 0, "cache_hits": 0}
    first = FieldObjective(full.model, full.encoder, full.obs, full.device, 4, np.arange(8), cache, stats)
    second = FieldObjective(full.model, full.encoder, full.obs, full.device, 4, [0, 7], cache, stats)
    full.model.query_sizes.clear()
    first([.31, .67])
    second([.31, .67])
    assert sum(full.model.query_sizes) == 3
    assert stats == {"parameter_predictions": 1, "field_queries": 3, "cache_hits": 1}
    for z in np.linspace(0, 1, 70):
        first([z, z])
    assert len(cache) == 64
    assert (.31, .67) not in cache


@pytest.mark.parametrize("indices", [[0, 0], [-1], [8], [1.5], []])
def test_invalid_sensor_indices_rejected(indices):
    full = linear_objective()
    with pytest.raises(ValueError, match="Sensor indices"):
        FieldObjective(full.model, full.encoder, full.obs, full.device, 4, indices)


def tiny_sweep_config(config, cases=1):
    return replace(config, inverse=InverseConfig(sweep_cases=cases, grid_size=3, batch_size=8,
                   max_iterations=2, max_evaluations=5, edge_max_evaluations=3))


def test_all_five_configurations_artifacts_and_baseline_equivalence(tmp_path, tiny_config, tiny_dataset, sweep_checkpoint):
    config = tiny_sweep_config(tiny_config)
    output = tmp_path / "sweep"
    report = sensor_sweep(config, tiny_dataset, sweep_checkpoint, output, device="cpu", verbose=False)
    assert report["failed_cases"] == []
    manifest = json.loads((output / "manifest.json").read_text())
    ids = manifest["selected_ids"]
    assert len(ids) == 1 and manifest["completed_configuration_cases"] == 5
    assert manifest["status"] == "completed"
    config_rows = {}
    for key, layout in manifest["sensor_layouts"].items():
        directory = output / "configurations" / key / "cases" / str(ids[0])
        for filename in ("grid.npz", "optimizer.json", "optimizer_trace.csv", "jacobians.npz", "fields.npz",
                         "metrics.json", "sensitivities.json", "result.json", "loss_surface.png", "reconstruction.png"):
            assert (directory / filename).stat().st_size > 0
        result = json.loads((directory / "result.json").read_text())
        config_rows[key] = result["rows"]
        assert {r["sim_id"] for r in result["rows"]} == set(ids)
        assert {r["observed_scalars"] for r in result["rows"]} == {5 * layout["sensor_count"] * 2}
        with np.load(directory / "jacobians.npz") as data:
            assert data["truth_physical"].shape == (5, layout["sensor_count"], 2, 2)
            assert len(data["x_centers"]) == layout["sensor_count"]
        metrics = json.loads((directory / "metrics.json").read_text())
        with np.load(directory / "fields.npz") as data:
            assert data["observed"].shape == (5, 64, 2)
            idx = data["sensor_indices"]
            for arm in ("4", "8"):
                expected = field_metrics(data[f"fv_{arm}"][:, idx], data["observed"][:, idx], data["times"])
                actual = metrics[arm]["sensor"]["fv_vs_observation"]
                for channel in ("S", "I"):
                    for metric in ("bias", "rmse", "relative_l2"):
                        assert actual[channel][metric] == pytest.approx(expected[channel][metric], abs=1e-15)
                    for a, b in zip(actual[channel]["by_time"], expected[channel]["by_time"]):
                        assert a == pytest.approx(b, abs=1e-15)
        with np.load(output / "configurations" / key / "validation_residuals.npz") as sparse:
            with np.load(output / "validation_residuals.npz") as full:
                assert np.array_equal(sparse["residuals"], full["residuals"][:, :, layout["indices"], :])
        optimizer = json.loads((directory / "optimizer.json").read_text())
        if key == "baseline":
            starts = [r["start"] for r in optimizer["starts"]][1:]
        else:
            assert [r["start"] for r in optimizer["starts"]][1:] == starts
            assert report["paired_changes"][key]["8"]["paired_count"] == 1
    for filename in ("cases.csv", "paired_changes.csv", "summary.json", "shared_costs.json",
                     "sensor_recovery.png", "sensor_replay_errors.png", "sensor_sensitivities.png", "sensor_start_stability.png"):
        assert (output / filename).stat().st_size > 0
    standalone = tmp_path / "standalone"
    inverse(config, tiny_dataset, sweep_checkpoint, standalone, sim_ids=ids, figures=False, device="cpu", verbose=False)
    original = json.loads((standalone / "cases" / str(ids[0]) / "result.json").read_text())["rows"]
    for a, b in zip(original, config_rows["baseline"]):
        for key in ("D_hat", "beta_hat", "loss", "condition_number", "column_cosine"):
            assert a[key] == pytest.approx(b[key], abs=1e-15)
        for channel in ("S", "I"):
            assert a[f"fv_vs_observation_{channel}_rmse"] == pytest.approx(b[f"fv_vs_observation_full_grid_{channel}_rmse"], abs=1e-15)
    baseline = config_rows["baseline"][1]
    for key in manifest["sensor_layouts"]:
        if key != "baseline":
            expected = config_rows[key][1]["D_abs_error"] - baseline["D_abs_error"]
            assert report["paired_changes"][key]["8"]["changes"]["D_abs_error_change"]["median"] == expected
    for path in output.rglob("*.json"):
        assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()


def test_shared_diagnostic_caches_avoid_repeated_fv_and_jacobians(tmp_path, tiny_config, tiny_dataset, sweep_checkpoint, monkeypatch):
    import fno_fox.inverse as module
    from fno_fox.data import FieldEncoder
    bundle = load_dataset(tiny_dataset)
    config = tiny_sweep_config(tiny_config)
    model, payload = load_checkpoint(sweep_checkpoint)
    encoder = FieldEncoder.from_dict(payload["encoder"])
    sim = int(bundle.splits["test"][0])
    counts = {"fv": 0, "jacobian": 0, "prediction": 0}
    fv, jac, prediction = module.fv_replay, FieldObjective.full_jacobians, FieldObjective.physical_prediction
    def fv_count(*args):
        counts["fv"] += 1
        return fv(*args)
    def jac_count(self, *args):
        counts["jacobian"] += 1
        return jac(self, *args)
    def prediction_count(self, *args):
        counts["prediction"] += 1
        return prediction(self, *args)
    monkeypatch.setattr(module, "fv_replay", fv_count)
    monkeypatch.setattr(FieldObjective, "full_jacobians", jac_count)
    monkeypatch.setattr(FieldObjective, "physical_prediction", prediction_count)
    shared = {}
    for iteration, idx in enumerate((np.arange(64), np.arange(64), np.array([0, 63]))):
        evaluate_case(bundle, sim, model, encoder, torch.device("cpu"), config.inverse, tmp_path / str(iteration),
                      sensor_indices=idx, shared=shared)
        if iteration == 0:
            previous = counts.copy()
        elif iteration == 1:
            assert counts == previous
    # Truth is shared regardless of the selected locations.
    assert counts["fv"] == len(shared["fv"])
    assert counts["jacobian"] == len(shared["jacobians"])
    assert counts["prediction"] == len(shared["predictions"])


def test_failures_pairing_overwrite_and_cli_exit(tmp_path, tiny_config, tiny_dataset, sweep_checkpoint, monkeypatch):
    import fno_fox.inverse_sensor_sweep as module
    config = tiny_sweep_config(tiny_config, cases=2)
    original = module.evaluate_case
    def sometimes_fail(*args, **kwargs):
        if kwargs["configuration"] == "sparse_c":
            raise ValueError("fixture failed sensor configuration")
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "evaluate_case", sometimes_fail)
    output = tmp_path / "sweep"
    report = sensor_sweep(config, tiny_dataset, sweep_checkpoint, output, verbose=False, figures=False)
    assert len(report["failed_cases"]) == 2
    assert report["configurations"]["sparse_c"]["completed_cases"] == 0
    assert report["paired_changes"]["sparse_c"]["8"]["paired_count"] == 0
    assert report["paired_changes"]["sparse_a"]["8"]["paired_count"] == 2
    with pytest.raises(FileExistsError):
        sensor_sweep(config, tiny_dataset, sweep_checkpoint, output, verbose=False)
    (output / "sensor_recovery.png").write_bytes(b"stale")
    sensor_sweep(config, tiny_dataset, sweep_checkpoint, output, overwrite=True, verbose=False, figures=False)
    assert not (output / "sensor_recovery.png").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "completed_with_failures"
    monkeypatch.setattr(module, "load_config", lambda p: config)
    assert main(["--config", "unused", "--data", str(tiny_dataset), "--checkpoint", str(sweep_checkpoint),
                 "--output", str(output), "--overwrite"]) == 1


def test_shared_grid_failure_preserves_case_records(tmp_path, tiny_config, tiny_dataset, sweep_checkpoint, monkeypatch):
    import fno_fox.inverse_sensor_sweep as module
    def fail(*args):
        raise ValueError("bad shared grid")
    monkeypatch.setattr(module, "shared_joint_grids", fail)
    output = tmp_path / "grid_failure"
    report = sensor_sweep(tiny_sweep_config(tiny_config), tiny_dataset, sweep_checkpoint, output,
                          verbose=False, figures=False)
    assert len(report["failed_cases"]) == 5
    for key in report["configurations"]:
        sim = report["selected_ids"][0]
        assert (output / "configurations" / key / "cases" / str(sim) / "failure.json").exists()


def test_output_guard_and_cli_selection(tmp_path, tiny_config, tiny_dataset, sweep_checkpoint, monkeypatch):
    import fno_fox.inverse_sensor_sweep as module
    config = tiny_sweep_config(tiny_config)
    with pytest.raises(ValueError, match="input"):
        sensor_sweep(config, tiny_dataset, sweep_checkpoint, tiny_dataset, overwrite=True)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    with pytest.raises(ValueError, match="existing fno_fox"):
        sensor_sweep(config, tiny_dataset, sweep_checkpoint, unrelated, overwrite=True)
    captured = {}
    monkeypatch.setattr(module, "load_config", lambda p: config)
    def run(*args, **kwargs):
        captured.update(kwargs)
        return {"failed_cases": []}
    monkeypatch.setattr(module, "sensor_sweep", run)
    assert main(["--config", "c", "--data", "d", "--checkpoint", "p", "--output", "o",
                 "--sim-ids", "3", "--device", "cpu"]) == 0
    assert captured["sim_ids"] == [3] and captured["device"] == "cpu"
    with pytest.raises(SystemExit):
        main(["--config", "c", "--data", "d", "--checkpoint", "p", "--output", "o",
              "--sim-ids", "3", "--all-test"])
