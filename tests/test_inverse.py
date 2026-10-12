from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fno_fox.config import InverseConfig, config_from_dict
from fno_fox.data import FieldEncoder, load_dataset
from fno_fox.inverse import (
    field_metrics, fv_replay, inverse, main, residual_summary,
    select_cases, validate_inputs,
)
from fno_fox.inverse_numerics import (
    FieldObjective, FullFieldObservations, joint_grid, observations, recover, sensitivity_summary,
)
from fno_fox.model import FNOConfig, build_model
from fno_fox.train import load_checkpoint, train


@pytest.fixture(scope="module", autouse=True)
def few_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def encoder():
    return FieldEncoder(0.8, 0.1, 0.1, 0.3, 10., 5., (.02, .08), (.6, 1.2))


class LinearFields(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.query_sizes = []

    def forward(self, spatial, cond):
        self.query_sizes.append(len(cond))
        x = spatial[..., 2]
        d, b, t = (cond[:, k, None] for k in range(3))
        return self.frozen_scale * torch.stack((d * (1 + x) * t + b * t,
                                                b * (2 - x) * t - d * t), dim=-1)


def linear_objective(batch=4, target_z=(.31, .67)):
    enc = encoder()
    source = np.column_stack((np.ones(8), np.zeros(8), np.linspace(0, 1, 8))).astype(np.float32)
    obs = FullFieldObservations(source, np.zeros((3, 8, 2), np.float32),
                                np.zeros((3, 8, 2)), np.array([.5, 1., 2.]))
    obj = FieldObjective(LinearFields(), enc, obs, torch.device("cpu"), batch)
    targets = obj.predict_many(np.array([target_z]))[0]
    obs = replace(obs, targets=targets, physical_targets=np.stack(enc.decode(targets), axis=-1))
    return FieldObjective(obj.model, enc, obs, torch.device("cpu"), batch)


def test_backward_compatible_config_and_invalid_budgets(default_config):
    old = default_config.as_dict()
    old.pop("inverse")
    assert config_from_dict(old).inverse == InverseConfig()
    for change in ({"grid_size": 2}, {"batch_size": 0}, {"start_seed": -1},
                   {"objective_tolerance": float("nan")}, {"max_evaluations": 0}):
        with pytest.raises(ValueError, match="inverse"):
            config_from_dict(old, {"inverse": change})


def test_selection_is_reproducible_balanced_and_test_only():
    bundle = SimpleNamespace(splits={"test": np.arange(4, 34)},
                             parameters={"susceptible_constant": np.arange(34) % 2 == 0})
    config = InverseConfig()
    first = select_cases(bundle, config)
    assert np.array_equal(first, select_cases(bundle, config))
    assert len(first) == 10
    assert bundle.parameters["susceptible_constant"][first].sum() == 5
    assert len(select_cases(bundle, config, all_test=True)) == 30
    assert np.array_equal(select_cases(bundle, config, [5, 4]), [4, 5])
    for args in (([1], False), ([4, 4], False), ([4], True)):
        with pytest.raises(ValueError):
            select_cases(bundle, config, *args)
    bundle.parameters["susceptible_constant"][:] = True
    assert len(select_cases(bundle, config)) == 10
    bundle.splits["test"] = np.array([4, 5, 6])
    assert len(select_cases(bundle, config)) == 3


def test_exact_observations_initial_source_and_no_truth(tiny_dataset):
    bundle = load_dataset(tiny_dataset)
    enc = FieldEncoder.from_bundle(bundle)
    sim = int(bundle.splits["test"][0])
    obs = observations(bundle, enc, sim)
    assert np.array_equal(obs.physical_targets[..., 0], bundle.susceptible[sim, 1:])
    assert np.array_equal(obs.physical_targets[..., 1], bundle.infectious[sim, 1:])
    assert np.array_equal(obs.times, bundle.times[1:])
    assert np.array_equal(obs.source, enc.spatial(bundle.susceptible[sim, 0], bundle.infectious[sim, 0], bundle.x_centers))
    assert obs.targets.shape == (bundle.n_snapshots - 1, len(bundle.x_centers), 2)
    assert not hasattr(obs, "theta_true")
    bundle.parameters["diffusivity"][sim] *= 1.1
    changed = observations(bundle, enc, sim)
    assert np.array_equal(changed.targets, obs.targets)
    assert np.array_equal(changed.source, obs.source)


def test_objective_batching_scaling_and_frozen_weights():
    obj = linear_objective(batch=2)
    original = {k: p.detach().clone() for k, p in obj.model.named_parameters()}
    points = np.array([[0, 0], [.31, .67], [1, 1]])
    batch = obj.predict_many(points)
    assert np.max(obj.model.query_sizes) <= 2
    assert batch.shape == (3, 3, 8, 2)
    assert np.allclose(obj.losses(points), [.5 * np.mean((p.astype(float) - obj.obs.targets) ** 2) for p in batch])
    assert obj(points[1]) == 0
    assert np.allclose(obj.physical_prediction(points[1]), obj.obs.physical_targets)
    obj.jacobians(points[1])
    assert not any(p.requires_grad for p in obj.model.parameters())
    assert all(torch.equal(original[k], p) for k, p in obj.model.named_parameters())
    assert not hasattr(obj, "theta_true")
    assert np.allclose(obj.physical_parameters([0, 1]), [.02, 1.2])


def test_grid_and_recovery_known_linear_operator():
    obj = linear_objective()
    grid = joint_grid(obj.losses, 9)
    result = recover(obj, grid, InverseConfig(), seed=0)
    assert np.allclose(result["arms"]["8"]["z"], [.31, .67], atol=2e-4)
    assert np.allclose(grid["axis"][[0, -1]], [0, 1])
    assert grid["evaluations"] == 81


@pytest.mark.parametrize("optimum", [(0.37, .61), (0., .413), (1., .6), (.4, 0.), (.4, 1.), (0., 1.), (1., 1.)])
def test_interior_edge_corner_and_budgeted_nested_starts(optimum):
    objective = lambda z: float(np.sum((np.asarray(z) - optimum) ** 2))
    grid = joint_grid(lambda pts: [objective(z) for z in pts], 7)
    config = InverseConfig()
    result = recover(objective, grid, config, 12)
    for count in ("4", "8"):
        assert np.allclose(result["arms"][count]["z"], optimum, atol=1e-4)
    assert len(result["starts"]) == 8
    assert len(result["edges"]) == 4
    assert all(r["evaluations"] <= 250 and r["iterations"] <= 100 for r in result["starts"])
    assert all(r["evaluations"] <= 50 for r in result["edges"])
    assert len(result["trace"]) == sum(r["evaluations"] for r in result["starts"] + result["edges"])
    for count in (4, 8):
        arm = result["arms"][str(count)]
        assert arm["evaluations"] == 49 + sum(r["evaluations"] for r in result["starts"][:count] + result["edges"])
    assert result["arms"]["8"]["loss"] <= result["arms"]["4"]["loss"]
    repeated = recover(objective, grid, config, 12)
    assert np.array_equal(np.array([r["start"] for r in result["starts"]]),
                          np.array([r["start"] for r in repeated["starts"]]))


def test_budget_exhaustion_not_reported_as_convergence():
    objective = lambda z: float(np.sum((np.asarray(z) - [.321, .678]) ** 2))
    grid = joint_grid(lambda pts: [objective(z) for z in pts], 3)
    result = recover(objective, grid, InverseConfig(max_iterations=1, max_evaluations=1, edge_max_evaluations=1), 0)
    assert all(not r["converged"] and r["budget_exhausted"] for r in result["starts"])
    assert result["arms"]["8"]["nonconverged_starts"] == 8
    assert all(r["evaluations"] <= 1 for r in result["starts"] + result["edges"])


@pytest.mark.parametrize("kind", ["valley", "multimodal"])
def test_narrow_valley_and_multiple_basins(kind):
    def objective(z):
        x, y = z
        if kind == "valley":
            return float((x + y - 1) ** 2 + 1e-3 * (x - .3) ** 2)
        return float(min((x - .2)**2 + (y - .3)**2 + .01, (x - .8)**2 + (y - .7)**2))
    grid = joint_grid(lambda pts: [objective(p) for p in pts], 25)
    result = recover(objective, grid, InverseConfig(), 0)
    assert result["arms"]["8"]["loss"] < 1e-9
    assert np.allclose(result["arms"]["8"]["z"], [.3, .7] if kind == "valley" else [.8, .7], atol=1e-3)


def test_linear_jacobian_and_physical_unit_conversion():
    obj = linear_objective(batch=2)
    physical, scaled = obj.jacobians([.4, .6])
    x, t = obj.obs.source[:, 2], obj.obs.times / obj.encoder.horizon
    normalized = np.empty((3, 8, 2, 2))
    normalized[:, :, 0, 0] = t[:, None] * (1 + x)
    normalized[:, :, 0, 1] = t[:, None]
    normalized[:, :, 1, 0] = -t[:, None]
    normalized[:, :, 1, 1] = t[:, None] * (2 - x)
    assert np.allclose(scaled, normalized.reshape(-1, 2) / np.sqrt(48))
    expected_physical = normalized * np.array([.1, .3])[None, None, :, None] / np.array([.06, .6])
    assert np.allclose(physical, expected_physical)
    changed = FieldObjective(obj.model, replace(obj.encoder, diffusivity_range=(20., 80.)), obj.obs, torch.device("cpu"), 3)
    different_units, same_scaled = changed.jacobians([.4, .6])
    assert np.allclose(same_scaled, scaled)
    assert np.allclose(different_units[..., 0] * 1000, physical[..., 0])


@pytest.mark.parametrize("point", [[.4, .6], [0., 1.]])
def test_fno_jvp_fft_matches_finite_differences(point):
    torch.manual_seed(6)
    obj = linear_objective(batch=2)
    model = build_model(FNOConfig(3, 4, 1, 8, 0.0))
    with torch.no_grad():
        model.cond_mlp.head.weight.normal_(0, .2)
    obj = FieldObjective(model, obj.encoder, obj.obs, torch.device("cpu"), 2)
    _, scaled = obj.jacobians(point)
    derivative = scaled * np.sqrt(scaled.shape[0])
    point = np.array(point)
    errors = []
    for h in (1e-2, 3e-3, 1e-3):
        cols = []
        for k in range(2):
            low, high = point.copy(), point.copy()
            if point[k] == 0:
                high[k] += h
            elif point[k] == 1:
                low[k] -= h
            else:
                low[k] -= h
                high[k] += h
            pair = obj.predict_many(np.array([low, high])).astype(float)
            cols.append(((pair[1] - pair[0]) / (high[k] - low[k])).ravel())
        approx = np.column_stack(cols)
        errors.append(np.linalg.norm(approx - derivative) / np.linalg.norm(derivative))
    assert max(errors) < .03


def test_svd_independent_collinear_zero_and_small_sensitivities():
    independent = sensitivity_summary(np.eye(2))
    assert independent["condition_number"] == 1
    assert independent["column_cosine"] == 0
    assert not independent["weakest_direction_unique"]
    collinear = sensitivity_summary(np.array([[1, -2], [2, -4], [3, -6]]))
    assert collinear["column_cosine"] == pytest.approx(-1)
    assert collinear["rank_deficient"]
    assert collinear["condition_number"] is None
    assert np.linalg.norm(np.array([[1, -2], [2, -4], [3, -6]]) @ collinear["weakest_direction"]) < 1e-12
    zero = sensitivity_summary(np.zeros((4, 2)))
    assert zero["numerical_rank"] == 0
    assert zero["column_cosine"] is None and zero["weakest_direction"] is None
    one = sensitivity_summary([[1, 0], [2, 0]])
    assert one["zero_columns"] == [False, True]
    small = sensitivity_summary(1e-12 * np.eye(2))
    assert small["condition_number"] == 1
    assert small["column_norms"][0] == 1e-12


def test_physical_metrics_and_structured_signed_residuals():
    truth = np.ones((2, 3, 2))
    prediction = truth + np.array([2., -1.])
    metrics = field_metrics(prediction, truth, np.array([1., 2.]))
    assert metrics["S"]["rmse"] == 2 and metrics["I"]["bias"] == -1
    assert metrics["S"]["relative_l2"] == 2
    assert metrics["I"]["by_time"][1]["rmse"] == 1
    assert field_metrics(truth, np.zeros_like(truth), [1, 2])["S"]["relative_l2"] is None
    residuals = np.stack([prediction - truth, 2 * (prediction - truth)])
    report, arrays = residual_summary(residuals)
    assert report["I"]["bias"] == -1.5
    assert report["S"]["rmse"] == pytest.approx(np.sqrt(10))
    assert np.allclose(arrays["signed_mean"][..., 0], 3)
    assert np.allclose(arrays["std_across_trajectories"][..., 0], 1)


def test_fv_replay_at_truth_and_only_disease_parameters_change(tiny_dataset, monkeypatch):
    import fno_fox.inverse as module
    from fno_fox.evaluate import trajectory_parameters
    bundle = load_dataset(tiny_dataset)
    sim = int(bundle.splits["test"][0])
    initial = trajectory_parameters(bundle, sim)
    truth = [initial.diffusivity, initial.transmission]
    replay = fv_replay(bundle, sim, truth)
    observations_ = np.stack([bundle.susceptible[sim, 1:], bundle.infectious[sim, 1:]], axis=-1)
    assert np.allclose(replay, observations_, atol=1e-7)
    original = module.simulate
    captured = []
    def capture(config, parameters, **kwargs):
        captured.append(parameters)
        return original(config, parameters, **kwargs)
    monkeypatch.setattr(module, "simulate", capture)
    fv_replay(bundle, sim, [.03, .8])
    assert captured == [replace(initial, diffusivity=.03, transmission=.8)]


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory, tiny_config, tiny_dataset):
    path = tmp_path_factory.mktemp("inverse_training") / "run"
    train(tiny_config, tiny_dataset, path, device="cpu", verbose=False)
    return path / "best.pt"


def test_provenance_rejections(tiny_config, tiny_dataset, tiny_checkpoint):
    bundle = load_dataset(tiny_dataset)
    _, payload = load_checkpoint(tiny_checkpoint)
    validate_inputs(tiny_config, bundle, payload)
    with pytest.raises(ValueError, match="provenance"):
        validate_inputs(tiny_config, bundle, {**payload, "dataset": {}})
    with pytest.raises(ValueError, match="normalization"):
        validate_inputs(tiny_config, bundle, {**payload, "encoder": {**payload["encoder"], "s_std": 99}})
    with pytest.raises(ValueError, match="domain"):
        validate_inputs(replace(tiny_config, domain=replace(tiny_config.domain, mortality=.9)), bundle, payload)
    bundle.splits["test"][0] = bundle.splits["train"][0]
    with pytest.raises(ValueError, match="partition"):
        validate_inputs(tiny_config, bundle, payload)


def test_tiny_inverse_end_to_end_and_artifacts(tmp_path, tiny_config, tiny_dataset, tiny_checkpoint):
    config = replace(tiny_config, inverse=InverseConfig(pilot_cases=1, grid_size=3, batch_size=8,
                     max_iterations=2, max_evaluations=5, edge_max_evaluations=3))
    out = tmp_path / "inverse"
    report = inverse(config, tiny_dataset, tiny_checkpoint, out, device="cpu", verbose=False)
    assert report["completed_cases"] == 1
    assert report["failed_cases"] == []
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert len(manifest["selected_ids"]) == 1
    assert set(manifest["selected_ids"]).isdisjoint(manifest["validation_ids"])
    case = out / "cases" / str(manifest["selected_ids"][0])
    for name in ("cases.csv", "summary.json", "validation_residuals.npz", "validation_residuals.json",
                 "recovery.png", "start_comparison.png", "sensitivities.png", "validation_residuals.png"):
        assert (out / name).stat().st_size > 0
    for name in ("grid.npz", "jacobians.npz", "fields.npz", "optimizer.json", "optimizer_trace.csv",
                 "sensitivities.json", "metrics.json", "result.json", "loss_surface.png", "reconstruction.png"):
        assert (case / name).stat().st_size > 0
    with np.load(case / "jacobians.npz") as jac:
        assert jac["truth_physical"].shape == (5, 64, 2, 2)
    with np.load(case / "fields.npz") as data:
        assert data["observed"].shape == data["fv_8"].shape == (5, 64, 2)
    optimizer = json.loads((case / "optimizer.json").read_text())
    saved_result = json.loads((case / "result.json").read_text())
    unique_estimates = {tuple(optimizer["arms"][arm]["z"]) for arm in ("4", "8")}
    assert saved_result["unique_fv_replays"] == len(unique_estimates)
    assert optimizer["seed"] == config.inverse.start_seed + manifest["selected_ids"][0]
    for path in out.rglob("*.json"):
        assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()
    with pytest.raises(FileExistsError):
        inverse(config, tiny_dataset, tiny_checkpoint, out, verbose=False)
    with pytest.raises(ValueError, match="input"):
        inverse(config, tiny_dataset, tiny_checkpoint, tiny_dataset, overwrite=True)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    with pytest.raises(ValueError, match="existing fno_fox"):
        inverse(config, tiny_dataset, tiny_checkpoint, unrelated, overwrite=True)


def test_case_failure_recorded_and_stale_artifacts_removed(tmp_path, tiny_config, tiny_dataset, tiny_checkpoint, monkeypatch):
    import fno_fox.inverse as module
    config = replace(tiny_config, inverse=InverseConfig(pilot_cases=1, grid_size=3))
    output = tmp_path / "failed"
    def fail(*args, **kwargs):
        raise ValueError("non-finite diagnostic fixture")
    monkeypatch.setattr(module, "evaluate_case", fail)
    # Driver must create the case folder even when a case fails immediately.
    report = inverse(config, tiny_dataset, tiny_checkpoint, output, verbose=False, figures=False)
    assert len(report["failed_cases"]) == 1
    assert report["completed_cases"] == 0
    assert json.loads((output / "manifest.json").read_text())["status"] == "completed_with_failures"
    (output / "cases.csv").write_text("stale results")
    (output / "recovery.png").write_bytes(b"stale figure")
    inverse(config, tiny_dataset, tiny_checkpoint, output, verbose=False, figures=False, overwrite=True)
    assert not (output / "cases.csv").exists()
    assert not (output / "recovery.png").exists()


def test_cli_dispatch_and_failure_exit_code(monkeypatch, tmp_path, tiny_config):
    import fno_fox.inverse as module
    captured = {}
    monkeypatch.setattr(module, "load_config", lambda p: tiny_config)
    def run(*args, **kwargs):
        captured.update(kwargs)
        return {"failed_cases": [{"sim_id": 3}]}
    monkeypatch.setattr(module, "inverse", run)
    assert main(["--config", "c", "--data", "d", "--checkpoint", "p", "--output", str(tmp_path),
                 "--sim-ids", "3", "4", "--device", "cpu"]) == 1
    assert captured["sim_ids"] == [3, 4] and captured["device"] == "cpu"
    with pytest.raises(SystemExit):
        main(["--config", "c", "--data", "d", "--checkpoint", "p", "--output", "o",
              "--sim-ids", "3", "--all-test"])
