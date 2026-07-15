import numpy as np

from src.priors import GWParameters
from src.waveform import (
    F_CROSS,
    F_PLUS,
    generate_waveform,
    inclination_amplitude_scale,
    N_SAMPLES,
)


def test_waveform_generation():
    """
    Basic sanity test for waveform simulator.
    """

    # --------------------------------------------------------
    # Fixed test parameters
    # --------------------------------------------------------

    theta = GWParameters(
        m1=40,
        m2=30,
        chi1=0.2,
        chi2=-0.1,
        distance=400,
        inclination=np.arccos(0.3),
    )

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

    assert np.isfinite([F_PLUS, F_CROSS]).all()


def test_inclination_amplitude_symmetry():
    """The single-detector amplitude is symmetric for iota and pi-iota."""

    inclination = 0.7
    assert np.isclose(
        inclination_amplitude_scale(inclination),
        inclination_amplitude_scale(np.pi - inclination),
    )


if __name__ == "__main__":
    test_waveform_generation()
    test_inclination_amplitude_symmetry()
    print("All waveform tests passed.")
