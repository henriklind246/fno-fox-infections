from __future__ import annotations

from collections import OrderedDict
import csv
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from fno_fox.config import InverseConfig, config_from_dict
from fno_fox.data import load_dataset
from fno_fox.generate import generate_dataset
from fno_fox.inverse import CHANNEL_SCOPE_LABELS, field_metrics, residual_summary, write_csv
from fno_fox.inverse_infected_sensor_sweep import (
    COMPARISONS, CONFIGURATIONS, infected_sensor_sweep, layout_key, main, observation_layouts,
)
from fno_fox.inverse_numerics import FieldObjective, joint_grid, recover, sensitivity_summary, shared_joint_grids
from fno_fox.inverse_sensor_sweep import select_sweep_cases
from fno_fox.inverse_sensor_time_sweep import nested_time_layouts, sensor_time_sweep
from fno_fox.train import train
from test_inverse import linear_objective
from test_inverse_sensor_sweep import fake_bundle


KEYS = [layout_key(n, channels) for n, channels in CONFIGURATIONS]
I_ONLY, BOTH = np.array([1]), np.array([0, 1])
ROOT_FIGURES = ("infected_sensor_recovery.png", "infected_sensor_channel_comparison.png",
                "infected_sensor_count_comparison.png", "infected_sensor_sensitivities.png",
                "infected_sensor_replay_errors.png", "infected_sensor_start_stability.png")


@pytest.fixture(scope="module", autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def channel_fixture(tmp_path_factory, tiny_config):
    # Six positive times, so the five selected times withhold one snapshot. The temporal
    # sweep's smallest count is also five, giving identical times for baseline comparison.
    config = config_from_dict(tiny_config.as_dict(), {
        "domain": {"n_snapshots": 7}, "training": {"epochs": 1},
        "inverse": {"sweep_cases": 1, "sweep_time_counts": [6, 5], "grid_size": 3,
                    "batch_size": 8, "max_iterations": 2, "max_evaluations": 5, "edge_max_evaluations": 3}})
    root = tmp_path_factory.mktemp("infected_sensor_fixture")
    generate_dataset(config, root / "data", verbose=False)
    train(config, root / "data", root / "run", device="cpu", verbose=False)
    return config, root / "data", root / "run" / "best.pt"


def test_five_layouts_in_table_order_share_five_times():
    times = np.arange(1, 21) / 4
    x = (np.arange(256) + .5) * 10 / 256
    layouts = observation_layouts(x, times)
    assert list(layouts) == ["sensors_32_times_5_channels_I", "sensors_16_times_5_channels_I",
                             "sensors_8_times_5_channels_I", "sensors_16_times_5_channels_SI",
                             "sensors_8_times_5_channels_SI"]
    expected_times = np.rint(np.linspace(0, 19, 5)).astype(int)
    # Identical to the five-time temporal-sweep layout at default counts.
    assert np.array_equal(expected_times, nested_time_layouts(times, (20, 15, 10, 7, 5))[5])
    for layout in layouts.values():
        n, channels = layout["sensor_count"], layout["observed_channels"]
        assert np.array_equal(layout["time_indices"], expected_times)
        assert np.array_equal(layout["times"], times[expected_times])
        assert np.array_equal(layout["indices"], np.rint(np.linspace(0, 255, n)).astype(int))
        assert np.array_equal(layout["x_centers"], x[layout["indices"]])
        assert np.array_equal(layout["channel_indices"], [("S", "I").index(c) for c in channels])
        assert layout["channel_label"] == "/".join(channels)
        assert layout["observed_scalars"] == n * 5 * len(channels)
        assert layout["channel_roles"] == {"S": "fitted" if "S" in channels else "evaluation_only", "I": "fitted"}
    assert [v["observed_scalars"] for v in layouts.values()] == [160, 80, 40, 160, 80]
    # Matched channel comparisons share positions; different counts are not nested.
    assert np.array_equal(layouts[KEYS[1]]["indices"], layouts[KEYS[3]]["indices"])
    assert np.array_equal(layouts[KEYS[2]]["indices"], layouts[KEYS[4]]["indices"])
    assert not set(layouts[KEYS[1]]["indices"]) <= set(layouts[KEYS[0]]["indices"])


def test_hundred_case_selection_and_times_independent_of_temporal_counts(tmp_path, default_config, monkeypatch):
    import fno_fox.inverse_infected_sensor_sweep as module
    bundle = fake_bundle()
    bundle.times = np.arange(21) / 4
    bundle.x_centers = np.arange(256)
    monkeypatch.setattr(module, "load_dataset", lambda p: bundle)
    captured = []
    def runner(config, actual_bundle, checkpoint, output, ids, layouts, **kwargs):
        captured.append({"ids": ids, "layouts": layouts, **kwargs})
        return {"failed_cases": []}
    monkeypatch.setattr(module, "_run_sweep", runner)
    other = config_from_dict(default_config.as_dict(), {"inverse": {"sweep_time_counts": [20, 3]}})
    for config in (default_config, other):
        infected_sensor_sweep(config, "unused", "unused", tmp_path / "output", figures=False)
    first, second = captured
    assert len(first["ids"]) == len(np.unique(first["ids"])) == 100
    assert bundle.parameters["susceptible_constant"][first["ids"]].sum() == 50
    assert np.array_equal(first["ids"], select_sweep_cases(bundle, default_config.inverse))
    assert list(first["layouts"]) == KEYS and len(first["ids"]) * len(first["layouts"]) == 500
    assert first["pipeline"] == "fno_fox.inverse_infected_sensor_sweep"
    for key in KEYS:
        assert np.array_equal(first["layouts"][key]["time_indices"], second["layouts"][key]["time_indices"])


@pytest.mark.parametrize("problem", ["cases", "cells", "times"])
def test_inadequate_inputs_fail_before_checkpoint_and_output(tmp_path, default_config, monkeypatch, problem):
    import fno_fox.inverse_infected_sensor_sweep as module
    bundle = fake_bundle(90 if problem == "cases" else 110)
    bundle.x_centers = np.arange(31 if problem == "cells" else 32)
    bundle.times = np.arange(5 if problem == "times" else 6) / 4
    monkeypatch.setattr(module, "load_dataset", lambda p: bundle)
    def should_not_run(*args, **kwargs):
        pytest.fail("Runner/model reached before layout or cardinality check")
    monkeypatch.setattr(module, "_run_sweep", should_not_run)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match={"cases": "100.*90", "cells": "32.*31", "times": "5.*4"}[problem]):
        infected_sensor_sweep(default_config, "unused", "unused.pt", output)
    assert not output.exists()


@pytest.mark.parametrize("indices", [[1, 1], [-1], [2], [.5], [], [1, 0], [True], [[1]]])
def test_invalid_objective_channel_indices(indices):
    full = linear_objective()
    with pytest.raises(ValueError, match="Channel indices"):
        FieldObjective(full.model, full.encoder, full.obs, full.device, 4, channel_indices=indices)


def test_explicit_both_channels_reproduce_default_objective_and_jacobians():
    full = linear_objective()
    sensors, times = [0, 3, 7], [0, 2]
    default = FieldObjective(full.model, full.encoder, full.obs, full.device, 4, sensors, time_indices=times)
    explicit = FieldObjective(full.model, full.encoder, full.obs, full.device, 4, sensors, time_indices=times,
                              channel_indices=BOTH)
    points = np.array([[0, 0], [.31, .67], [1, 1]])
    assert np.array_equal(default.losses(points), explicit.losses(points))
    for a, b in zip(default.jacobians([.4, .6]), explicit.jacobians([.4, .6])):
        assert np.array_equal(a, b)


def test_later_time_s_targets_cannot_change_infected_only_fitting():
    full = linear_objective()
    sensors, times = np.array([0, 3, 7]), np.arange(3)

    def objectives(obs):
        return [FieldObjective(full.model, full.encoder, obs, full.device, 4, sensors, time_indices=times,
                               channel_indices=channels) for channels in (I_ONLY, BOTH)]

    normalized, physical = full.obs.targets.copy(), full.obs.physical_targets.copy()
    normalized[1:, :, 0] += 5
    physical[1:, :, 0] += 5 * full.encoder.s_std
    changed = replace(full.obs, targets=normalized, physical_targets=physical)
    (infected, both), (infected_changed, both_changed) = objectives(full.obs), objectives(changed)
    points = np.array([[0, 0], [.31, .67], [.5, .5], [1, 1]])
    assert np.array_equal(infected.losses(points), infected_changed.losses(points))
    assert np.all(both_changed.losses(points) > both.losses(points))
    config = InverseConfig(max_iterations=5, max_evaluations=12, edge_max_evaluations=5)
    a, b = [recover(obj, joint_grid(obj.losses, 3), config, 0) for obj in (infected, infected_changed)]
    for arm in ("4", "8"):
        assert np.array_equal(a["arms"][arm]["z"], b["arms"][arm]["z"])
        assert a["arms"][arm]["loss"] == b["arms"][arm]["loss"]
    c, d = [recover(obj, joint_grid(obj.losses, 3), config, 0) for obj in (both, both_changed)]
    assert d["arms"]["8"]["loss"] > c["arms"]["8"]["loss"]
    # Half mean square of I residuals alone, in the checkpoint's I normalization.
    residual = (full.predict_many([[.5, .5]])[0].astype(float) - full.obs.targets)[times][:, sensors][..., 1]
    assert infected([.5, .5]) == pytest.approx(.5 * np.mean(residual ** 2), rel=1e-12)


def test_shared_channel_grids_match_direct_losses_and_share_full_predictions():
    full = linear_objective(batch=2)
    selection = {"indices": np.array([0, 7]), "time_indices": np.array([0, 2])}
    layouts = {"infected": {**selection, "channel_indices": I_ONLY},
               "both": {**selection, "channel_indices": BOTH}, "legacy": selection}
    full.model.query_sizes.clear()
    grids, cost = shared_joint_grids(full, layouts, 5)
    assert sum(full.model.query_sizes) == 25 * 3 and cost["field_queries"] == 75
    cache, stats = OrderedDict(), {"parameter_predictions": 0, "field_queries": 0, "cache_hits": 0}
    for key, layout in layouts.items():
        obj = FieldObjective(full.model, full.encoder, full.obs, full.device, 2, layout["indices"], cache, stats,
                             time_indices=layout["time_indices"], channel_indices=layout.get("channel_indices"))
        assert np.array_equal(grids[key]["loss"], joint_grid(obj.losses, 5)["loss"])
        obj([.31, .67])
    assert np.array_equal(grids["both"]["loss"], grids["legacy"]["loss"])
    assert not np.array_equal(grids["infected"]["loss"], grids["both"]["loss"])
    # Only the first objective queries the model (25 grid points and one optimizer point);
    # later channel layouts reuse the cached complete S/I predictions.
    assert stats["parameter_predictions"] == 26 and stats["field_queries"] == 26 * 3
    assert cache[(.31, .67)].shape == (1, 3, 8, 2)


@pytest.mark.parametrize("point", [[.4, .6], [0., 1.]])
def test_infected_jacobian_selection_range_scaling_and_finite_differences(point):
    full = linear_objective(batch=2)
    sensors, times = np.array([0, 3, 7]), np.array([0, 2])
    infected = FieldObjective(full.model, full.encoder, full.obs, full.device, 2, sensors, time_indices=times,
                              channel_indices=I_ONLY)
    physical, scaled = infected.jacobians(point)
    full_physical, full_scaled = full.jacobians(point)
    m = 2 * 3 * 1
    assert physical.shape == (2, 3, 1, 2) and scaled.shape == (m, 2)
    assert np.array_equal(physical, full_physical[times][:, sensors][:, :, [1]])
    expected = full_scaled.reshape(3, 8, 2, 2)[times][:, sensors][:, :, [1]].reshape(-1, 2) * np.sqrt(48 / m)
    assert np.array_equal(scaled, expected)
    # diag(1 / I scale) * J_physical * diag(parameter ranges) / sqrt(m)
    assert np.allclose(scaled.reshape(physical.shape), physical / full.encoder.i_std * full.ranges / np.sqrt(m))
    assert sensitivity_summary(scaled)["observed_scalars"] == m
    for h in (1e-2, 3e-3, 1e-3):
        derivatives = []
        for k in range(2):
            lo, hi = np.array(point), np.array(point)
            lo[k], hi[k] = max(0, lo[k] - h), min(1, hi[k] + h)
            values = infected.observe(full.predict_many(np.array([lo, hi])).astype(float))
            derivatives.append(((values[1] - values[0]) / (hi[k] - lo[k])).ravel())
        assert np.allclose(np.column_stack(derivatives) / np.sqrt(m), scaled, rtol=2e-4, atol=1e-5)


def test_channel_subset_metrics_residual_summaries_and_union_csv(tmp_path):
    truth = np.ones((2, 3, 1))
    metrics = field_metrics(truth + 2, truth, [1., 2.], ("I",))
    assert list(metrics) == ["I"] and metrics["I"]["rmse"] == 2 and len(metrics["I"]["by_time"]) == 2
    report, arrays = residual_summary(np.full((2, 2, 3, 1), -1.), ["I"])
    assert list(report) == ["I"] and report["I"]["bias"] == -1 and arrays["rmse"].shape == (2, 3, 1)
    write_csv(tmp_path / "rows.csv", [{"a": 1, "b": 2}, {"a": 3, "c": 4}])
    assert (tmp_path / "rows.csv").read_text().splitlines() == ["a,b,c", "1,2,", "3,,4"]


def assert_metrics(actual, expected):
    assert list(actual) == list(expected)
    for channel in expected:
        for metric in ("bias", "rmse", "relative_l2"):
            assert actual[channel][metric] == pytest.approx(expected[channel][metric], rel=1e-12, abs=1e-15)
        assert len(actual[channel]["by_time"]) == len(expected[channel]["by_time"])
        for a, b in zip(actual[channel]["by_time"], expected[channel]["by_time"]):
            assert a == pytest.approx(b, rel=1e-12, abs=1e-15)


def test_five_configuration_fixture_artifacts_pairing_and_recomputed_baselines(tmp_path, channel_fixture):
    config, data, checkpoint = channel_fixture
    root = tmp_path / "sweep"
    report = infected_sensor_sweep(config, data, checkpoint, root, device="cpu", verbose=False)
    assert report["failed_cases"] == [] and report["completed_configuration_cases"] == 5
    manifest = json.loads((root / "manifest.json").read_text())
    sim = manifest["selected_ids"][0]
    assert manifest["status"] == "completed" and list(manifest["observation_layouts"]) == KEYS
    assert manifest["observed_scalars_by_configuration"] == dict(zip(KEYS, [160, 80, 40, 160, 80]))
    assert manifest["full_field_channels"] == ["S", "I"]
    assert manifest["scheduled_ids_by_configuration"] == {key: [sim] for key in KEYS}
    std = np.array([manifest["encoder"]["s_std"], manifest["encoder"]["i_std"]])
    ranges = np.diff(manifest["parameter_bounds"], axis=1).ravel()
    with np.load(root / "validation_residuals.npz") as shared:
        full_residuals = shared["residuals"]
    rows, starts = {}, None
    for key, layout in manifest["observation_layouts"].items():
        directory = root / "configurations" / key / "cases" / str(sim)
        for name in ("grid.npz", "optimizer.json", "optimizer_trace.csv", "jacobians.npz", "fields.npz",
                     "metrics.json", "sensitivities.json", "result.json", "loss_surface.png", "reconstruction.png"):
            assert (directory / name).stat().st_size > 0
        for name in ("cases.csv", "summary.json", "recovery.png", "sensitivities.png", "validation_residuals.png"):
            assert (root / "configurations" / key / name).stat().st_size > 0
        channels, n, m = layout["observed_channels"], layout["sensor_count"], layout["observed_scalars"]
        case_rows = rows[key] = json.loads((directory / "result.json").read_text())["rows"]
        assert {r["observed_scalars"] for r in case_rows} == {m}
        assert {r["observed_channels"] for r in case_rows} == {layout["channel_label"]}
        assert ("fno_vs_observation_sensor_S_rmse" in case_rows[1]) == ("S" in channels)
        assert "fno_vs_observation_full_grid_S_rmse" in case_rows[1]
        optimizer = json.loads((directory / "optimizer.json").read_text())
        current_starts = [r["start"] for r in optimizer["starts"]][1:]
        starts = current_starts if starts is None else starts
        assert current_starts == starts and optimizer["seed"] == config.inverse.start_seed + sim
        with np.load(directory / "jacobians.npz") as jac:
            assert list(jac["channel_names"]) == channels
            assert np.array_equal(jac["channel_indices"], layout["channel_indices"])
            assert np.array_equal(jac["times"], layout["times"])
            for label in ("truth", "grid", "starts_4", "starts_8"):
                assert jac[f"{label}_physical"].shape == (5, n, len(channels), 2)
                assert jac[f"{label}_scaled"].shape == (m, 2)
            scale = std[layout["channel_indices"]]
            assert np.allclose(jac["truth_scaled"].reshape(jac["truth_physical"].shape),
                               jac["truth_physical"] / scale[:, None] * ranges / np.sqrt(m))
        sensitivities = json.loads((directory / "sensitivities.json").read_text())
        assert sensitivities["truth"]["observed_scalars"] == m
        assert sensitivities["truth"]["observed_channels"] == channels
        singular = sensitivities["starts_8"]["singular_values"]
        assert sensitivities["starts_8"]["singular_value_ratio"] == pytest.approx(singular[1] / singular[0])
        assert case_rows[1]["singular_value_ratio"] == sensitivities["starts_8"]["singular_value_ratio"]
        assert [case_rows[1]["weakest_direction_D"], case_rows[1]["weakest_direction_beta"]] == \
            sensitivities["starts_8"]["weakest_direction"]
        metrics = json.loads((directory / "metrics.json").read_text())
        assert metrics["channel_roles"] == layout["channel_roles"]
        assert metrics["scope_labels"] == CHANNEL_SCOPE_LABELS
        with np.load(directory / "fields.npz") as fields:
            assert fields["observed"].shape == fields["fno_8"].shape == (6, 64, 2)
            assert np.array_equal(fields["observed_channel_indices"], layout["channel_indices"])
            assert list(fields["field_channel_names"]) == ["S", "I"]
            ti, si, ci = fields["time_indices"], fields["sensor_indices"], fields["observed_channel_indices"]
            for arm in ("4", "8"):
                fno, fv, truth = fields[f"fno_{arm}"], fields[f"fv_{arm}"], fields["observed"]
                for comparison, a, b in (("fno_vs_observation", fno, truth), ("fv_vs_observation", fv, truth),
                                         ("fno_vs_fv", fno, fv)):
                    assert_metrics(metrics[arm]["sensor"][comparison],
                                   field_metrics(a[ti][:, si][..., ci], b[ti][:, si][..., ci], fields["times"][ti], channels))
                    assert_metrics(metrics[arm]["full_grid_selected_times"][comparison],
                                   field_metrics(a[ti], b[ti], fields["times"][ti]))
                    assert_metrics(metrics[arm]["full_grid"][comparison], field_metrics(a, b, fields["times"]))
        with np.load(root / "configurations" / key / "validation_residuals.npz") as residuals:
            expected = full_residuals[:, layout["time_indices"]][:, :, layout["indices"]][..., layout["channel_indices"]]
            assert np.array_equal(residuals["residuals"], expected)
            assert list(residuals["channel_names"]) == channels
    for name in ("cases.csv", "summary.json", "paired_changes.csv", "shared_costs.json", *ROOT_FIGURES):
        assert (root / name).stat().st_size > 0
    with (root / "cases.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 10
    for row in table:
        assert (row["fno_vs_observation_sensor_S_rmse"] != "") == row["configuration"].endswith("_SI")
    assert set(report["paired_changes"]) == {"channel", "sensor_count", "equal_scalar_budget"}
    for kind, key, reference in COMPARISONS:
        comparison = report["paired_changes"][kind][key]["8"]
        assert comparison["reference_configuration"] == reference
        assert comparison["paired_count"] == 1 and comparison["unpaired_ids"] == []
        for metric in ("D_abs_error", "beta_error", "singular_value_1", "singular_value_ratio",
                       "fv_vs_observation_full_grid_S_rmse", "fno_vs_observation_sensor_I_rmse"):
            expected = rows[key][1][metric] - rows[reference][1][metric]
            assert comparison["changes"][f"{metric}_change"]["median"] == pytest.approx(expected, abs=1e-15)
        # Sensor-scope S exists only for S/I; differently normalized losses are never paired.
        assert "fno_vs_observation_sensor_S_rmse_change" not in comparison["changes"]
        assert not any(metric.startswith("loss") for metric in comparison["changes"])
        assert 0 <= comparison["changes"]["weakest_direction_angle_deg"]["median"] <= 90
        assert comparison["optimizer"]["current"]["nonconverged_starts"] == rows[key][1]["nonconverged_starts"]
        assert comparison["optimizer"]["reference"]["nonconverged_starts"] == rows[reference][1]["nonconverged_starts"]
    with (root / "paired_changes.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2 * len(COMPARISONS)
    costs = json.loads((root / "shared_costs.json").read_text())["cases"][str(sim)]
    assert costs["grid"]["parameter_predictions"] == 9 and costs["grid"]["field_queries"] == 54
    assert costs["diagnostics"]["unique_jacobian_points"] < 5 * 4
    # S/I configurations are recomputed here and match the temporal sweep's five-time results.
    temporal = tmp_path / "temporal"
    sensor_time_sweep(config, data, checkpoint, temporal, sim_ids=[sim], device="cpu", figures=False, verbose=False)
    for n in (16, 8):
        old = json.loads((temporal / "configurations" / f"sensors_{n}_times_5" / "cases" / str(sim)
                          / "result.json").read_text())["rows"]
        for a, b in zip(old, rows[layout_key(n, ("S", "I"))]):
            for metric in ("D_hat", "beta_hat", "loss", "singular_value_1", "condition_number", "column_cosine",
                           "fv_vs_observation_sensor_S_rmse", "fv_vs_observation_full_grid_I_rmse",
                           "fno_vs_fv_full_grid_selected_times_S_rmse"):
                assert a[metric] == pytest.approx(b[metric], abs=1e-15)
    for path in root.rglob("*.json"):
        assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()


def test_failures_paired_exclusions_figures_overwrite_and_nonzero_exit(tmp_path, channel_fixture, monkeypatch):
    import fno_fox.inverse_infected_sensor_sweep as module
    import fno_fox.inverse_sensor_sweep as engine
    config, data, checkpoint = channel_fixture
    config = replace(config, inverse=replace(config.inverse, sweep_cases=2))
    failing = layout_key(16, ("S", "I"))
    original = engine.evaluate_case
    def sometimes_fail(*args, **kwargs):
        if kwargs["configuration"] == failing:
            raise RuntimeError("Fixture failed S/I baseline")
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, "evaluate_case", sometimes_fail)
    root = tmp_path / "failures"
    report = infected_sensor_sweep(config, data, checkpoint, root, device="cpu", verbose=False)
    ids = report["selected_ids"]
    assert len(report["failed_cases"]) == 2 and report["configurations"][failing]["completed_cases"] == 0
    for kind, key in (("channel", layout_key(16, ("I",))), ("equal_scalar_budget", layout_key(32, ("I",)))):
        comparison = report["paired_changes"][kind][key]["8"]
        assert comparison["paired_count"] == 0 and comparison["changes"] == {}
        assert comparison["unpaired_ids"] == comparison["failed_unpaired_ids"] == ids
    assert report["paired_changes"]["channel"][layout_key(8, ("I",))]["8"]["paired_count"] == 2
    assert report["paired_changes"]["sensor_count"][layout_key(16, ("I",))]["8"]["paired_count"] == 2
    for sim in ids:
        directory = root / "configurations" / failing / "cases" / str(sim)
        assert (directory / "failure.json").exists() and not (directory / "result.json").exists()
    for name in ROOT_FIGURES:
        assert (root / name).stat().st_size > 0
    with pytest.raises(FileExistsError):
        infected_sensor_sweep(config, data, checkpoint, root, figures=False)
    stale = root / "infected_sensor_channel_comparison.png"
    stale.write_bytes(b"stale")
    monkeypatch.setattr(module, "load_config", lambda p: config)
    assert main(["--config", "unused", "--data", str(data), "--checkpoint", str(checkpoint),
                 "--output", str(root), "--device", "cpu", "--overwrite"]) == 1
    assert stale.read_bytes() != b"stale"
    assert json.loads((root / "manifest.json").read_text())["status"] == "completed_with_failures"


def test_shared_grid_failure_marks_all_five_configuration_cases(tmp_path, channel_fixture, monkeypatch):
    import fno_fox.inverse_sensor_sweep as engine
    config, data, checkpoint = channel_fixture
    def fail(*args):
        raise ValueError("Fixture shared grid failure")
    monkeypatch.setattr(engine, "shared_joint_grids", fail)
    root = tmp_path / "grid_failure"
    report = infected_sensor_sweep(config, data, checkpoint, root, figures=False, verbose=False)
    assert report["failed_configuration_cases"] == 5 and report["completed_configuration_cases"] == 0
    for key in KEYS:
        assert (root / "configurations" / key / "cases" / str(report["selected_ids"][0]) / "failure.json").exists()


def test_capped_resume_reuses_successes_and_checks_channel_layouts(tmp_path, channel_fixture, monkeypatch):
    import fno_fox.inverse_sensor_sweep as engine
    config, data, checkpoint = channel_fixture
    config = replace(config, inverse=replace(config.inverse, sweep_cases=2))
    ids = sorted(map(int, load_dataset(data).splits["test"][:2]))
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
        infected_sensor_sweep(config, data, checkpoint, root, device="cpu", sim_ids=ids, figures=False, verbose=False)
    assert calls == [(ids[0], key) for key in KEYS[:3]]
    saved = {path: path.read_bytes() for path in root.rglob("result.json")}
    assert len(saved) == 3
    manifest_path = root / "manifest.json"
    before = manifest_path.read_bytes()

    def resume():
        return infected_sensor_sweep(config, data, checkpoint, root, device="cpu", sim_ids=ids, figures=False,
                                     verbose=False, resume=True, max_recovery_problems=7)

    tampered = json.loads(before)
    tampered["sensor_layouts"][KEYS[0]]["channel_indices"] = [0, 1]
    manifest_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="sensor_layouts"):
        resume()
    manifest_path.write_bytes(before)
    result_path = root / "configurations" / KEYS[0] / "cases" / str(ids[0]) / "result.json"
    for field, value in (("observed_scalars", 320), ("observed_channels", "S/I")):
        payload = json.loads(saved[result_path])
        payload["rows"][0][field] = value
        result_path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="Invalid saved recovery result"):
            resume()
        assert manifest_path.read_bytes() == before
    result_path.write_bytes(saved[result_path])
    calls.clear()
    def record(*args, **kwargs):
        calls.append((args[1], kwargs["configuration"]))
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, "evaluate_case", record)
    report = resume()
    assert calls == [(ids[0], KEYS[3]), (ids[0], KEYS[4]), (ids[1], KEYS[0]), (ids[1], KEYS[1])]
    assert report["completed_configuration_cases"] == 7 and report["budget_excluded_configuration_cases"] == 3
    for path, content in saved.items():
        assert path.read_bytes() == content
    manifest = json.loads(manifest_path.read_text())
    assert manifest["resume"]["reused_configuration_cases"] == 3
    for key, scheduled in manifest["scheduled_ids_by_configuration"].items():
        assert scheduled == (ids if key in KEYS[:2] else ids[:1])
    channel = report["paired_changes"]["channel"][KEYS[1]]["8"]
    assert channel["paired_count"] == 1 and channel["unpaired_ids"] == [ids[1]] and channel["failed_unpaired_ids"] == []
    assert report["paired_changes"]["sensor_count"][KEYS[1]]["8"]["paired_count"] == 2
    calls.clear()
    resume()
    assert calls == []


def test_cli_options_and_output_guards(tmp_path, channel_fixture, monkeypatch):
    import fno_fox.inverse_infected_sensor_sweep as module
    config, data, checkpoint = channel_fixture
    with pytest.raises(ValueError, match="input"):
        infected_sensor_sweep(config, data, checkpoint, data, overwrite=True, figures=False)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    with pytest.raises(ValueError, match="existing fno_fox.inverse_infected_sensor_sweep"):
        infected_sensor_sweep(config, data, checkpoint, unrelated, overwrite=True, figures=False)
    wrong_pipeline = tmp_path / "temporal"
    wrong_pipeline.mkdir()
    (wrong_pipeline / "manifest.json").write_text(json.dumps({"pipeline": "fno_fox.inverse_sensor_time_sweep"}))
    with pytest.raises(ValueError, match="inverse_infected_sensor_sweep"):
        infected_sensor_sweep(config, data, checkpoint, wrong_pipeline, resume=True, figures=False)
    monkeypatch.setattr(module, "load_config", lambda p: config)
    captured = {}
    def run(*args, **kwargs):
        captured.update(kwargs)
        return {"failed_cases": []}
    monkeypatch.setattr(module, "infected_sensor_sweep", run)
    assert main(["--config", "c", "--data", "d", "--checkpoint", "p", "--output", "o",
                 "--device", "cpu", "--sim-ids", "1", "--overwrite"]) == 0
    assert captured["device"] == "cpu" and captured["sim_ids"] == [1] and captured["overwrite"]
    assert main(["--config", "c", "--data", "d", "--checkpoint", "p", "--output", "o", "--all-test"]) == 0
    assert captured["all_test"]
    assert main(["--config", "c", "--data", "d", "--checkpoint", "p", "--output", "o",
                 "--resume", "--max-recovery-problems", "500"]) == 0
    assert captured["resume"] and captured["max_recovery_problems"] == 500
    for conflicting in (["--all-test", "--sim-ids", "1"], ["--resume", "--overwrite"]):
        with pytest.raises(SystemExit):
            main(["--config", "c", "--data", "d", "--checkpoint", "p", "--output", "o", *conflicting])


@pytest.mark.parametrize("budget", [0, -1, True, 2.5, 6])
def test_invalid_recovery_budget_before_output(tmp_path, channel_fixture, budget):
    config, data, checkpoint = channel_fixture
    root = tmp_path / "invalid"
    with pytest.raises(ValueError, match="max_recovery_problems"):
        infected_sensor_sweep(config, data, checkpoint, root, figures=False, max_recovery_problems=budget)
    assert not root.exists()
