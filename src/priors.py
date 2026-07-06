from dataclasses import dataclass
import numpy as np


@dataclass
class GWParameters:
    """
    Container for one gravitational-wave parameter sample.
    """

    m1: float
    m2: float
    chi1: float
    chi2: float
    distance: float
    inclination: float

    def __str__(self):
        """Pretty print the parameter values."""
        return (
            "\n"
            "=====================================================\n"
            "        Gravitational Wave Parameters (θ)\n"
            "=====================================================\n"
            f" Primary Mass (m1)      : {self.m1:8.2f} M☉\n"
            f" Secondary Mass (m2)    : {self.m2:8.2f} M☉\n"
            f" Primary Spin (χ1)      : {self.chi1:8.4f}\n"
            f" Secondary Spin (χ2)    : {self.chi2:8.4f}\n"
            f" Luminosity Distance    : {self.distance:8.2f} Mpc\n"
            f" Inclination (ι)        : {self.inclination:8.4f} rad\n"
            "====================================================="
        )
    
    def to_array(self):
        """
        Convert the parameter object into a NumPy array.

        Returns
        -------
        np.ndarray
            Shape (6,)
        """

        return np.array([
            self.m1,
            self.m2,
            self.chi1,
            self.chi2,
            self.distance,
            self.inclination,
        ], dtype=np.float32)

class PriorSampler:
    """
    Samples parameters from the prior distributions.
    """

    def __init__(
        self,
        mass_range=(10.0, 80.0),
        spin_range=(-0.99, 0.99),
        distance_range=(100.0, 600.0),
    ):
        self.mass_range = mass_range
        self.spin_range = spin_range
        self.distance_range = distance_range

    def sample_masses(self):
        """
        Sample component masses while enforcing m1 >= m2.
        """

        m1 = np.random.uniform(*self.mass_range)
        m2 = np.random.uniform(*self.mass_range)

        if m2 > m1:
            m1, m2 = m2, m1

        return m1, m2

    def sample_spins(self):
        """
        Sample aligned spin components.
        """

        chi1 = np.random.uniform(*self.spin_range)
        chi2 = np.random.uniform(*self.spin_range)

        return chi1, chi2

    def sample_distance(self):
        """
        Sample luminosity distance (Mpc).
        """

        return np.random.uniform(*self.distance_range)

    def sample_inclination(self):
        """
        Sample inclination angle.

        The sampling is uniform in cos(iota), not in iota.
        """

        cos_iota = np.random.uniform(-1.0, 1.0)
        return np.arccos(cos_iota)

    def sample(self):
        """
        Sample one complete gravitational-wave parameter vector.
        """

        m1, m2 = self.sample_masses()
        chi1, chi2 = self.sample_spins()
        distance = self.sample_distance()
        inclination = self.sample_inclination()

        return GWParameters(
            m1=m1,
            m2=m2,
            chi1=chi1,
            chi2=chi2,
            distance=distance,
            inclination=inclination,
        )
