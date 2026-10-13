from copy import deepcopy

import numpy as np
import pytest

from fno_fox.data import FieldEncoder, load_dataset
from fno_fox.extend_test import extend_test_dataset, validate_test_extension
from fno_fox.inverse import validate_inputs


@pytest.fixture
def extended(tmp_path, tiny_dataset):
    path = extend_test_dataset(tiny_dataset, tmp_path / "extension", count=4, seed=6, verbose=False)
    return load_dataset(path)


def checkpoint_payload(source):
    return {"config": source.manifest["config"], "encoder": FieldEncoder.from_bundle(source).as_dict(),
            "dataset": {key: source.manifest[key] for key in ("dataset_sha256", "config_sha256")}}


def test_fresh_test_data_preserves_checkpoint_inputs(extended, tiny_dataset):
    source = load_dataset(tiny_dataset)
    assert extended.n_trajectories == source.n_trajectories + 4
    added = np.array(extended.manifest["test_extension"]["added_test_ids"])
    assert len(added) == len(np.unique(added)) == 4
    assert extended.parameters["susceptible_constant"][added].sum() == 2
    assert np.isin(added, extended.splits["test"]).all()
    assert not np.isin(added, np.concatenate((extended.splits["train"], extended.splits["val"]))).any()
    assert np.array_equal(extended.susceptible[:source.n_trajectories], source.susceptible)
    assert extended.normalization == source.normalization
    validate_inputs(extended.config, extended, checkpoint_payload(source))
    assert extended.manifest["solver_diagnostics"]["maximum_relative_balance_residual"] < 1e-10
    with pytest.raises(FileExistsError):
        extend_test_dataset(tiny_dataset, extended.root, count=4, verbose=False)


@pytest.mark.parametrize("change", ["fields", "parameters", "split", "provenance", "physics"])
def test_reject_changed_original_data(extended, change):
    if change == "fields":
        extended.susceptible[0, 0, 0] += .1
    elif change == "parameters":
        extended.parameters["diffusivity"][0] += .01
    elif change == "split":
        extended.splits["train"] = extended.splits["train"][::-1]
    elif change == "provenance":
        extended.manifest["test_extension"]["dataset_sha256"] = "incorrect"
    else:
        extended.manifest["config"]["domain"]["mortality"] += .1
    with pytest.raises(ValueError, match="Test extension"):
        validate_test_extension(extended)


def test_extension_does_not_accept_unrelated_checkpoint(extended, tiny_dataset):
    payload = checkpoint_payload(load_dataset(tiny_dataset))
    payload["dataset"]["dataset_sha256"] = "unrelated"
    with pytest.raises(ValueError, match="provenance"):
        validate_inputs(extended.config, extended, payload)
    # An extension marker is required; unrelated datasets remain rejected.
    no_marker = deepcopy(extended)
    no_marker.manifest.pop("test_extension")
    with pytest.raises(ValueError):
        validate_inputs(no_marker.config, no_marker, checkpoint_payload(load_dataset(tiny_dataset)))
