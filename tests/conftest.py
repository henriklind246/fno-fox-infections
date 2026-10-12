from __future__ import annotations

from pathlib import Path

import pytest

from fno_fox.config import FoxConfig, load_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "conf" / "default.yaml"

# Small but structurally complete experiment: same physics and pair rules as
# the default, fewer trajectories/snapshots and a tiny model.
TINY_OVERRIDES = {
    "domain": {"horizon": 1.0, "n_snapshots": 6, "learning_cells": 64},
    "sampling": {"n_trajectories": 20},
    "refinement": {"random_pilots": 0},
    "pairs": {"later_source_per_trajectory": 5},
    "model": {"modes": 8, "width": 8, "n_layers": 2, "cond_hidden": 16},
    "training": {"epochs": 2, "batch_size": 16, "device": "cpu"},
    "evaluation": {
        "batch_size": 64,
        "speed_window": [0.4, 1.0],
        "min_front_points": 2,
        "interpolation_trajectories": 2,
        "interpolation_source_times": [0.0, 0.6],
        "representative_trajectories": 2,
    },
}


@pytest.fixture(scope="session")
def default_config() -> FoxConfig:
    return load_config(DEFAULT_CONFIG)


@pytest.fixture(scope="session")
def tiny_config() -> FoxConfig:
    return load_config(DEFAULT_CONFIG, TINY_OVERRIDES)


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory, tiny_config):
    from fno_fox.generate import generate_dataset

    root = tmp_path_factory.mktemp("tiny_dataset")
    generate_dataset(tiny_config, root, verbose=False)
    return root
