"""
BayesFlow training helpers for gravitational-wave source inference.

This module expects an offline dataset saved as an ``.npz`` file with:

    X     -> whitened noisy strain, shape (n_simulations, n_samples)
    theta -> true parameters, shape (n_simulations, 6)

Parameter order is fixed as:
    [m1, m2, chi1, chi2, distance, inclination]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


PARAMETER_NAMES = ("m1", "m2", "chi1", "chi2", "distance", "inclination")
DEFAULT_DATASET_PATH = Path("data/gw_dataset_1000.npz")
DEFAULT_MODEL_DIR = Path("models/bayesflow_model")


def _require_bayesflow():
    """
    Import BayesFlow lazily so dataset checks can run without ML dependencies.
    """

    os.environ.setdefault("KERAS_BACKEND", "torch")

    try:
        import bayesflow as bf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "BayesFlow is not installed. Install the project requirements first, "
            "for example: pip install -r requirements.txt"
        ) from exc

    return bf


def validate_dataset_arrays(X: np.ndarray, theta: np.ndarray) -> None:
    """
    Validate the Person 2 dataset contract before training.
    """

    if X.ndim != 2:
        raise ValueError(f"X must have shape (n_simulations, n_samples), got {X.shape}")

    if theta.ndim != 2 or theta.shape[1] != len(PARAMETER_NAMES):
        raise ValueError(
            "theta must have shape (n_simulations, 6) with columns "
            f"{PARAMETER_NAMES}, got {theta.shape}"
        )

    if X.shape[0] != theta.shape[0]:
        raise ValueError(
            f"X and theta must contain the same number of simulations, got {X.shape[0]} and {theta.shape[0]}"
        )

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")

    if not np.isfinite(theta).all():
        raise ValueError("theta contains NaN or infinite values")

    if not np.all(theta[:, 0] >= theta[:, 1]):
        raise ValueError("Dataset violates the required mass ordering m1 >= m2")


def load_npz_dataset(path: str | Path, max_samples: int | None = None) -> dict[str, np.ndarray]:
    """
    Load and validate a GW dataset in the project-standard format.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Expected a .npz file containing arrays named X and theta."
        )

    with np.load(path) as data:
        missing = {"X", "theta"} - set(data.files)
        if missing:
            raise KeyError(f"Dataset {path} is missing required arrays: {sorted(missing)}")

        X = np.asarray(data["X"], dtype=np.float32)
        theta = np.asarray(data["theta"], dtype=np.float32)

    if max_samples is not None:
        X = X[:max_samples]
        theta = theta[:max_samples]

    validate_dataset_arrays(X, theta)
    return {"strain": X[..., None], "parameters": theta}


def split_dataset(
    dataset: dict[str, np.ndarray],
    validation_fraction: float = 0.1,
    seed: int = 2026,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Shuffle and split a dataset into train and validation dictionaries.
    """

    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")

    n_simulations = dataset["parameters"].shape[0]
    if n_simulations < 10:
        raise ValueError("Need at least 10 simulations for a train/validation split")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_simulations)
    n_val = max(1, int(round(validation_fraction * n_simulations)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_data = {key: value[train_idx] for key, value in dataset.items()}
    val_data = {key: value[val_idx] for key, value in dataset.items()}
    return train_data, val_data


def build_adapter() -> Any:
    """
    Build the BayesFlow adapter for offline GW datasets.

    The long strain time series is routed through ``summary_variables`` so it can
    be embedded by the summary network before conditioning the posterior network.
    """

    bf = _require_bayesflow()
    return bf.approximators.ContinuousApproximator.build_adapter(
        inference_variables="parameters",
        inference_conditions=None,
        summary_variables="strain",
    )


def build_workflow(model_dir: str | Path | None = None) -> Any:
    """
    Create a BayesFlow BasicWorkflow with a time-series summary network.
    """

    bf = _require_bayesflow()

    adapter = build_adapter()
    summary_network = bf.networks.TimeSeriesNetwork(
        kernel_sizes=2,
        recurrent_dim=64,
        skip_steps=1,
    )
    inference_network = bf.networks.CouplingFlow()

    checkpoint_filepath = None
    if model_dir is not None:
        checkpoint_filepath = str(Path(model_dir))

    return bf.BasicWorkflow(
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
        checkpoint_filepath=checkpoint_filepath,
        checkpoint_name="model",
        inference_variables="parameters",
        summary_variables="strain",
        standardize="all",
    )


def train_workflow(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    max_samples: int | None = None,
    epochs: int = 20,
    batch_size: int = 64,
    validation_fraction: float = 0.1,
    seed: int = 2026,
) -> tuple[Any, Any, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Load data, train the BayesFlow model offline, and save the workflow.
    """

    dataset = load_npz_dataset(dataset_path, max_samples=max_samples)
    train_data, val_data = split_dataset(dataset, validation_fraction=validation_fraction, seed=seed)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    workflow = build_workflow(model_dir=model_dir)
    history = workflow.fit_offline(
        data=train_data,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
    )

    return workflow, history, train_data, val_data


def load_workflow(model_dir: str | Path = DEFAULT_MODEL_DIR) -> Any:
    """
    Load a previously saved BayesFlow workflow.
    """

    _require_bayesflow()
    import keras

    model_dir = Path(model_dir)
    model_path = model_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved BayesFlow approximator not found: {model_path}")

    workflow = build_workflow(model_dir=model_dir)
    workflow.approximator = keras.saving.load_model(model_path)
    return workflow


def sample_posterior(
    workflow: Any,
    strain: np.ndarray,
    num_samples: int = 1_000,
) -> np.ndarray:
    """
    Draw posterior samples for one whitened strain signal.
    """

    strain = np.asarray(strain, dtype=np.float32)
    if strain.ndim == 1:
        strain = strain[None, :, None]
    elif strain.ndim == 2:
        strain = strain[..., None]
    else:
        raise ValueError(f"Expected strain with 1 or 2 dimensions, got shape {strain.shape}")

    posterior = workflow.sample(
        conditions={"strain": strain},
        num_samples=num_samples,
    )
    return np.asarray(posterior["parameters"][0])


def history_to_losses(history: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Extract train and validation losses from a Keras/BayesFlow history object.
    """

    history_dict = getattr(history, "history", history)
    if not isinstance(history_dict, dict) or "loss" not in history_dict:
        raise ValueError("Could not find a 'loss' series in the training history")

    train_loss = np.asarray(history_dict["loss"], dtype=float)
    val_loss = history_dict.get("val_loss")
    if val_loss is not None:
        val_loss = np.asarray(val_loss, dtype=float)

    return train_loss, val_loss
