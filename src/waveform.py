import numpy as np
from pycbc.waveform import get_td_waveform

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

DURATION = 4.0  # seconds
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

F_PLUS = 0.8
F_CROSS = 0.6
# Single detector simplification
# Keep fixed across all simulations
# simulation choice (engineering decision) 

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

def generate_waveform(theta):
    """
   → This function:
   maps theta (physical parameters) → clean gravitational-wave strain

    → It is used as the forward simulator in BayesFlow
    to generate synthetic training data for neural posterior estimation

    Generate clean gravitational-wave strain.

    Parameters
    ----------
    theta : dict
        Dictionary containing:
            m1
            m2
            chi1
            chi2
            distance
            cos_iota (indicated as cos function in the project documentation)

    Returns
    -------
    strain : np.ndarray
        Fixed-length strain array of shape (N_SAMPLES,)
    """

    # --------------------------------------------------------
    # Extract parameters
    # --------------------------------------------------------

    m1 = theta["m1"]
    m2 = theta["m2"]
    # Component masses of the binary black hole system
    # The waveform’s frequency evolution is entirely determined by these parameters

    chi1 = theta["chi1"]
    chi2 = theta["chi2"]
    # Spin parameters of the black holes
    # Affect the phase evolution of the gravitational waveform

    distance = theta["distance"]
    # Controls amplitude scaling (distance effect)
    # As distance increases, the gravitational-wave signal becomes weaker

    cos_iota = theta["cos_iota"]
    # Cosine form of the inclination angle (indacted in the project documentation)
    # Using a uniform prior in cos(iota) is a more physically meaningful choice
    # because it corresponds to isotropic orientation in 3D space 
    inclination = np.arccos(cos_iota)
    # → PyCBC requires the angle in radians

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
    strain = np.array(strain)
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