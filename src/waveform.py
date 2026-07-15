import numpy as np
from pycbc.detector import Detector
from pycbc.waveform import get_td_waveform
from src.priors import GWParameters

# ============================================================
# SIMULATION CONFIGURATION
# ============================================================


SAMPLING_RATE = 2048
# → Number of samples we take per second (Hz)
# → GW signals are continuous in time, but we discretize them for computation
# → 2048 Hz is a standard sampling rate close to LIGO conventions


DELTA_T = 1.0 / SAMPLING_RATE
# → Time step between consecutive samples (seconds)
# → Required by PyCBC waveform generation to define the time resolution

DURATION = 1.0  # seconds
# → Duration of the waveform (seconds)
# → ML for fixed-length input requirement

N_SAMPLES = int(SAMPLING_RATE * DURATION)
# → Total number of data points in the signal
# → BayesFlow requires a fixed input shape for neural network compatibility
# → Ensures consistent tensor dimensions across all simulations

F_LOWER = 20.0  # Hz
# → Minimum frequency used in waveform generation
# → Frequencies below 20 Hz are dominated by LIGO detector noise, so they are cut off

APPROXIMANT = "IMRPhenomD"
# → Physical waveform model used for binary black hole systems
# → Includes inspiral, merger, and ringdown phases of the coalescence
# → Aligned-spin approximation assumes spins are aligned with orbital angular momentum


# ============================================================
# FIXED DETECTOR RESPONSE
# ============================================================

DETECTOR_NAME = "H1"
RIGHT_ASCENSION = 1.0  # radians
DECLINATION = 0.5  # radians
POLARIZATION = 0.3  # radians
GPS_TIME = 1126259462.4

# The sky position, polarization and time are fixed for every simulation, as
# required by the project setup. The response factors are calculated once from
# the Hanford detector geometry instead of being chosen by hand.
_DETECTOR = Detector(DETECTOR_NAME)
F_PLUS, F_CROSS = _DETECTOR.antenna_pattern(
    RIGHT_ASCENSION,
    DECLINATION,
    POLARIZATION,
    GPS_TIME,
)


def inclination_amplitude_scale(inclination):
    """Return the detector amplitude factor associated with inclination.

    This quantity makes the distance--inclination degeneracy explicit: the
    observed amplitude is approximately proportional to this scale divided by
    luminosity distance. It is also symmetric under iota -> pi - iota.
    """

    cos_iota = np.cos(inclination)
    plus_factor = 0.5 * (1.0 + cos_iota**2)
    cross_factor = cos_iota
    return np.sqrt(
        (F_PLUS * plus_factor) ** 2 + (F_CROSS * cross_factor) ** 2
    )

# ============================================================
# HELPER: FIXED-LENGTH WINDOW EXTRACTION
# ============================================================

def crop_or_pad(signal, start, target_length):
    """
    Extract a fixed-length segment from a signal.

    If the requested window exceeds the signal boundaries,
    zero-padding is applied to maintain a consistent length.

    This function is used to convert variable-length PyCBC outputs
    into fixed-size waveforms required by machine learning models.

    PyCBC waveforms can naturally have varying lengths depending on
    physical parameters, but neural networks (e.g., BayesFlow)
    require a fixed input shape for batch processing and training stability.
    """

    end = start + target_length
    # → End point of the extraction window

    # --------------------------------------------------------
    # LEFT PADDING (if data is missing at the beginning)
    # --------------------------------------------------------
    if start < 0:
        left_pad = abs(start)
    else:
        left_pad = 0

    # --------------------------------------------------------
    # RIGHT PADDING (if data is missing at the end)
    # --------------------------------------------------------
    if end > len(signal):
        right_pad = end - len(signal)
    else:
        right_pad = 0

    # --------------------------------------------------------
    # padding process
    # --------------------------------------------------------
    if left_pad > 0 or right_pad > 0:
        signal = np.pad(signal, (left_pad, right_pad))
    # → Missing regions are filled with zeros
    # → Because there is no physical signal outside the observed waveform range

    # Shift indices after padding
    start += left_pad
    end = start + target_length

    # → Now we extract a fixed-length slice from the signal
    return signal[start:end]


# ============================================================
# MAIN WAVEFORM GENERATOR
# ============================================================

def generate_waveform(theta: GWParameters) -> np.ndarray:
    """
   → This function:
   maps theta (physical parameters) → clean gravitational-wave strain

    → It is used as the forward simulator in BayesFlow
    to generate synthetic training data for neural posterior estimation

    Generate clean gravitational-wave strain.

Parameters
----------
theta : GWParameters
    One sampled set of gravitational-wave parameters.

    Attributes
    ----------
    m1 : float
        Primary black hole mass (M☉)

    m2 : float
        Secondary black hole mass (M☉)

    chi1 : float
        Dimensionless aligned spin of the primary black hole

    chi2 : float
        Dimensionless aligned spin of the secondary black hole

    distance : float
        Luminosity distance (Mpc)

    inclination : float
        Inclination angle (radians)

Returns
-------
strain : np.ndarray
    Fixed-length strain array of shape (N_SAMPLES,)
    """

    # --------------------------------------------------------
    # Extract parameters
    # --------------------------------------------------------

    m1 = theta.m1
    m2 = theta.m2
    # Component masses of the binary black hole system

    chi1 = theta.chi1
    chi2 = theta.chi2
    # Dimensionless aligned spins

    distance = theta.distance
    # Luminosity distance (Mpc)

    inclination = theta.inclination
    # Inclination angle (already sampled in priors.py)

    # --------------------------------------------------------
    # Enforce physical ordering (indicated in the project documentation)
    if m1 < m2:
        raise ValueError("Require m1 >= m2")

    # --------------------------------------------------------
    # Waveform Generation
    # --------------------------------------------------------

    hp, hc = get_td_waveform(
        approximant=APPROXIMANT,
        mass1=m1,
        mass2=m2,
        spin1z=chi1,
        spin2z=chi2,
        distance=distance,
        inclination=inclination,
        delta_t=DELTA_T,
        f_lower=F_LOWER
    )

    # → hp = plus polarization
    # → hc = cross polarization
    # → Output of the general relativity (GR) waveform solution

    # --------------------------------------------------------
    # Detector projection
    # --------------------------------------------------------

    strain = F_PLUS * hp + F_CROSS * hc
    # → The real detector mixes the two polarization modes
    # → This produces a single-channel observed signal (indicated in the project documentation)


    # Convert to numpy
    strain = strain.numpy()
     # → PyCBC time series → numpy array (ML friendly)

    # --------------------------------------------------------
    # Locate merger peak (MERGER ALIGNMENT)
    # --------------------------------------------------------

    peak = np.argmax(np.abs(strain))
    # → The merger is the point where the signal reaches maximum amplitude

    # --------------------------------------------------------
    # Align merger near the end of the window
    # --------------------------------------------------------
    # Why?
    # Preserves inspiral information for ML inference.

    merger_position = int(0.7 * N_SAMPLES)
    # Determines where the merger is placed within the fixed-length waveform
    # The 0.7 factor ensures a longer inspiral phase is preserved for better inference

    start = peak - merger_position
    # → We start the waveform from this computed index


    # --------------------------------------------------------
    # Fixed-length crop/pad
    # --------------------------------------------------------

    strain = crop_or_pad(
        signal=strain,
        start=start,
        target_length=N_SAMPLES
    )
    # → Required for machine learning: ensures fixed-length input

    # --------------------------------------------------------
    # Apply taper window (WINDOWING)
    # --------------------------------------------------------
    # Prevents sharp discontinuities at boundaries.

    window = np.hanning(N_SAMPLES)
    # → Applied to prevent sharp discontinuities at the signal boundaries
    # → Reduces spectral artifacts in the frequency domain

    strain = strain * window

    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    if len(strain) != N_SAMPLES:
        raise RuntimeError("Waveform has incorrect length")
    # → Safety check to prevent dataset corruption_

    return strain
