import numpy as np

from pycbc.types import TimeSeries
from pycbc.psd import aLIGOZeroDetHighPower


class Whitening:
    """
    PSD-based whitening using the Advanced LIGO design sensitivity.

    This implementation whitens the signal using the same PSD
    that is used to generate detector noise.
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
        self.delta_f = 1.0 / duration
        self.n_samples = int(duration * sampling_rate)

        # Same PSD used by NoiseGenerator
        self.psd = aLIGOZeroDetHighPower(
            self.n_samples // 2 + 1,
            self.delta_f,
            self.f_lower,
        )

    def whiten(self, signal):
        """
        Whiten a noisy strain signal.

        Parameters
        ----------
        signal : np.ndarray
            Noisy strain.

        Returns
        -------
        np.ndarray
            Whitened strain.
        """

        ts = TimeSeries(signal, delta_t=self.delta_t)

        # Fourier transform
        hf = ts.to_frequencyseries()

        # Avoid division by zero
        psd = self.psd.copy()
        psd.data[psd.data == 0] = np.inf

        # Whitening
        hf /= np.sqrt(psd)

        # Back to time domain
        whitened = hf.to_timeseries()

        return np.array(whitened)