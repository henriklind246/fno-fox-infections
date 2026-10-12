from __future__ import annotations

from dataclasses import replace
import importlib
import json

import numpy as np
import pytest
import torch

from fno_fox.data import FieldEncoder, load_dataset
from fno_fox.model import FNOConfig, build_model
from fno_fox.train import save_checkpoint


evaluation = importlib.import_module("fno_fox.evaluate")


@pytest.fixture
def checkpoint(tmp_path, tiny_dataset, tiny_config):
    bundle = load_dataset(tiny_dataset)
    model_config = FNOConfig.from_model_config(tiny_config.model)
    model = build_model(model_config)
    path = tmp_path / "model.pt"
    save_checkpoint(
        path, model, model_config, FieldEncoder.from_bundle(bundle), tiny_config,
        epoch=0, metrics={},
    )
    return path


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_interpolation_uses_requested_split(
    tmp_path, tiny_dataset, tiny_config, checkpoint, monkeypatch, split
) -> None:
    bundle = load_dataset(tiny_dataset)
    regenerated = []
    original = evaluation.trajectory_parameters

    def record_parameters(bundle, sim):
        regenerated.append(sim)
        return original(bundle, sim)

    monkeypatch.setattr(evaluation, "trajectory_parameters", record_parameters)
    report = evaluation.evaluate(
        tiny_config, tiny_dataset, checkpoint, tmp_path / split,
        split=split, device="cpu", figures=False,
    )
    expected = np.sort(bundle.splits[split])[:tiny_config.evaluation.interpolation_trajectories].tolist()
    assert regenerated == expected
    assert report["interpolation"]["trajectories"] == expected
    assert report["interpolation"]["split"] == split


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_predictions_abort_evaluation(
    tmp_path, tiny_dataset, tiny_config, checkpoint, bad_value
) -> None:
    payload = torch.load(checkpoint, weights_only=False)
    payload["model_state"]["output_layer.bias"].fill_(bad_value)
    torch.save(payload, checkpoint)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="Non-finite model prediction.*batch starting at 0"):
        evaluation.evaluate(
            tiny_config, tiny_dataset, checkpoint, output, device="cpu", figures=False
        )
    assert not (output / "metrics.json").exists()
    assert not (output / "fronts.csv").exists()


def test_missing_front_speeds_are_null_in_strict_json(
    tmp_path, tiny_dataset, tiny_config, checkpoint
) -> None:
    # No trajectory can have this many crossings, guaranteeing missing speeds
    # independently of the model's random predictions.
    config = replace(tiny_config, evaluation=replace(
        tiny_config.evaluation, min_front_points=tiny_config.domain.n_snapshots + 1,
    ))
    output = tmp_path / "strict_json"
    report = evaluation.evaluate(
        config, tiny_dataset, checkpoint, output, device="cpu", figures=True
    )

    def reject_non_json_constant(value):
        raise AssertionError(f"Non-JSON constant: {value}")

    stored = json.loads(
        (output / "metrics.json").read_text(), parse_constant=reject_non_json_constant
    )
    assert stored == report
    assert all(row["true_speed"] is None and row["predicted_speed"] is None
               for row in stored["fronts"]["speeds"])
    assert stored["fronts"]["speed_abs_error_mean"] is None
    assert (output / "front_position.png").exists()


@pytest.mark.parametrize("source_time", [1.0, 0.35, float("nan")])
def test_invalid_interpolation_sources_fail_before_inference(
    tmp_path, tiny_dataset, tiny_config, monkeypatch, source_time
) -> None:
    config = replace(tiny_config, evaluation=replace(
        tiny_config.evaluation, interpolation_source_times=(source_time,),
    ))

    def unexpected_work(*args, **kwargs):
        pytest.fail("Invalid interpolation source must be rejected before model or solver work")

    monkeypatch.setattr(evaluation, "load_checkpoint", unexpected_work)
    monkeypatch.setattr(evaluation, "simulate", unexpected_work)
    with pytest.raises(ValueError, match="Interpolation source time"):
        evaluation.evaluate(config, tiny_dataset, tmp_path / "unused.pt", tmp_path / "output")

    # Direct callers receive the same validation before any reference solves.
    with pytest.raises(ValueError, match="Interpolation source time"):
        evaluation.interpolation_check(
            config, load_dataset(tiny_dataset), None, None, torch.device("cpu")
        )
