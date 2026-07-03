# tests/test_waveform.py

import numpy as np
import matplotlib.pyplot as plt

from waveform import (
    generate_waveform,
    N_SAMPLES,
    SAMPLING_RATE
)


def test_waveform_generation():
    """
    Basic sanity test for waveform simulator.
    """

    # --------------------------------------------------------
    # Fixed test parameters
    # --------------------------------------------------------

    theta = {
        "m1": 40,
        "m2": 30,
        "chi1": 0.2,
        "chi2": -0.1,
        "distance": 400,
        "cos_iota": 0.3
    }

    # --------------------------------------------------------
    # Generate waveform
    # --------------------------------------------------------

    strain = generate_waveform(theta)

    # --------------------------------------------------------
    # TEST 1: correct shape
    # --------------------------------------------------------

    assert strain.shape == (N_SAMPLES,), \
        f"Expected shape {(N_SAMPLES,)}, got {strain.shape}"

    # --------------------------------------------------------
    # TEST 2: NaN check
    # --------------------------------------------------------

    assert not np.isnan(strain).any(), \
        "Waveform contains NaN values"

    # --------------------------------------------------------
    # TEST 3: finite values
    # --------------------------------------------------------

    assert np.isfinite(strain).all(), \
        "Waveform contains non-finite values"

    # --------------------------------------------------------
    # TEST 4: non-zero signal
    # --------------------------------------------------------

    assert np.max(np.abs(strain)) > 0, \
        "Waveform is completely zero"

    print("All waveform tests passed.")

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    # Normalize ONLY for plotting

    plot_strain = strain / np.max(np.abs(strain))

    time = np.arange(N_SAMPLES) / SAMPLING_RATE

    plt.figure(figsize=(12, 4))

    plt.plot(time, plot_strain)

    plt.title("Test Gravitational Waveform")

    plt.xlabel("Time [s]")
    plt.ylabel("Normalized strain")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    test_waveform_generation()