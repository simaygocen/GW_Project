import numpy as np

from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd


class NoiseGenerator:
    """
    Generates colored Gaussian detector noise using
    the Advanced LIGO design sensitivity PSD.
    """

    def __init__(
        self,
        sampling_rate=2048,
        duration=4.0,
        f_lower=20.0,
    ):

        self.sampling_rate = sampling_rate
        self.duration = duration
        self.f_lower = f_lower

        self.delta_t = 1.0 / sampling_rate
        self.n_samples = int(duration * sampling_rate)
        self.delta_f = 1.0 / duration

        # Advanced LIGO Design PSD
        self.psd = aLIGOZeroDetHighPower(
            self.n_samples // 2 + 1,
            self.delta_f,
            self.f_lower,
        )

    def generate(self):
        """
        Generate one realization of colored detector noise.

        Returns
        -------
        noise : np.ndarray
            Shape (n_samples,)
        """

        noise = noise_from_psd(
            self.n_samples,
            self.delta_t,
            self.psd,
            seed=None,
        )

        return np.array(noise)

    def add_noise(self, waveform):
        """
        Add detector noise to a clean waveform.
        """

        noise = self.generate()

        return waveform + noise