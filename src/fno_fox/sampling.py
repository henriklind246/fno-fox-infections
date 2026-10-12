"""Scalar and functional sampling for fox-rabies trajectories.

Reduced from no-tps-ihcp ``src/physics/init_conditions.py`` (analytic,
grid-independent profile builders with explicit samplers) and
``data/generate_dataset.py`` (stratified Latin-hypercube parameters and
balanced family assignment) to the 1D distributions used here.

Initial profiles are evaluated as exact cell averages, so any refined solver
grid averages conservatively onto identical learning-grid initial fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.special import erf

from fno_fox.config import SamplingConfig


SUSCEPTIBLE_FAMILIES = ("constant", "cosine")


@dataclass(frozen=True)
class TrajectoryParameters:
    diffusivity: float
    transmission: float
    infection_amplitude: float
    infection_center: float
    infection_width: float
    susceptible_family: str
    susceptible_amplitude: float
    susceptible_coefficients: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["susceptible_coefficients"] = list(self.susceptible_coefficients)
        return payload


def latin_hypercube(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """One sample per stratum in every dimension, independently permuted."""
    edges = np.linspace(0.0, 1.0, n_samples + 1)
    unit = edges[:-1, None] + np.diff(edges)[:, None] * rng.random((n_samples, n_dims))
    for column in range(n_dims):
        rng.shuffle(unit[:, column])
    return unit


def balanced_assignment(
    n_samples: int,
    fraction_first: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Boolean mask with round(fraction * n) True entries in random order."""
    count = int(round(fraction_first * n_samples))
    mask = np.zeros(n_samples, dtype=bool)
    mask[:count] = True
    rng.shuffle(mask)
    return mask


def normalized_cosine_coefficients(n_modes: int, rng: np.random.Generator) -> np.ndarray:
    """Normal draws scaled to sum |c_k| = 1."""
    for _ in range(100):
        draws = rng.standard_normal(n_modes)
        total = float(np.sum(np.abs(draws)))
        if total > 0.0:
            return draws / total
    raise RuntimeError("Could not draw a nonzero cosine-coefficient vector.")


def sample_parameters(
    config: SamplingConfig,
    rng: np.random.Generator,
) -> list[TrajectoryParameters]:
    n = config.n_trajectories
    unit = latin_hypercube(n, 2, rng)
    d_low, d_high = config.diffusivity
    b_low, b_high = config.transmission
    diffusivity = d_low + unit[:, 0] * (d_high - d_low)
    transmission = b_low + unit[:, 1] * (b_high - b_low)
    amplitude = rng.uniform(*config.infection_amplitude, size=n)
    center = rng.uniform(*config.infection_center, size=n)
    width = rng.uniform(*config.infection_width, size=n)
    constant = balanced_assignment(n, config.susceptible_constant_fraction, rng)
    parameters = []
    for index in range(n):
        if constant[index]:
            family = "constant"
            perturbation = 0.0
            coefficients = np.zeros(config.susceptible_modes)
        else:
            family = "cosine"
            perturbation = float(rng.uniform(*config.susceptible_perturbation_amplitude))
            coefficients = normalized_cosine_coefficients(config.susceptible_modes, rng)
        parameters.append(
            TrajectoryParameters(
                diffusivity=float(diffusivity[index]),
                transmission=float(transmission[index]),
                infection_amplitude=float(amplitude[index]),
                infection_center=float(center[index]),
                infection_width=float(width[index]),
                susceptible_family=family,
                susceptible_amplitude=perturbation,
                susceptible_coefficients=tuple(float(value) for value in coefficients),
            )
        )
    return parameters


def gaussian_cell_average(
    faces: np.ndarray,
    amplitude: float,
    center: float,
    width: float,
) -> np.ndarray:
    """Exact cell averages of A exp(-(x - x_c)^2 / (2 sigma^2))."""
    scale = np.sqrt(2.0) * width
    integral = 0.5 * np.sqrt(np.pi) * scale * erf((faces - center) / scale)
    return amplitude * np.diff(integral) / np.diff(faces)


def cosine_cell_average(
    faces: np.ndarray,
    length: float,
    amplitude: float,
    coefficients: np.ndarray | tuple[float, ...],
) -> np.ndarray:
    """Exact cell averages of 1 + a sum_k c_k cos(k pi x / L)."""
    values = np.ones(len(faces) - 1)
    widths = np.diff(faces)
    for mode, coefficient in enumerate(coefficients, start=1):
        wavenumber = mode * np.pi / length
        values += (
            amplitude
            * coefficient
            * np.diff(np.sin(wavenumber * faces))
            / (wavenumber * widths)
        )
    return values


def initial_fields(
    parameters: TrajectoryParameters,
    faces: np.ndarray,
    length: float,
) -> tuple[np.ndarray, np.ndarray]:
    if parameters.susceptible_family not in SUSCEPTIBLE_FAMILIES:
        raise ValueError(f"Unknown susceptible family {parameters.susceptible_family!r}.")
    susceptible = cosine_cell_average(
        faces,
        length,
        parameters.susceptible_amplitude,
        parameters.susceptible_coefficients,
    )
    infectious = gaussian_cell_average(
        faces,
        parameters.infection_amplitude,
        parameters.infection_center,
        parameters.infection_width,
    )
    return susceptible, infectious
