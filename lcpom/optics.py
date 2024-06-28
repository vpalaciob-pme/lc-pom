from dataclasses import dataclass
import numpy as np

class TransmissionMode:
    """
    Base class representing a transmission mode for optics.
    """
    pass


class FullTransmission(TransmissionMode):
    """
    Represents full transmission where the transmittance is unchanged.

    Methods
    -------
    transmittance(T, *_args)
        Returns the transmittance T unchanged.
    
    Parameters
    ----------
    T : float
        Initial transmittance value.
    """
    def transmittance(self, T, *_args):
        return T


@dataclass
class Fresnel(TransmissionMode):
    """
    Represents Fresnel transmission mode for optics calculations.

    Attributes
    ----------
    n_medium : float
        Refractive index of the medium.

    Methods
    -------
    transmittance(T, lc, no, ne, ind)
        Calculates and returns the adjusted transmittance based on Fresnel equations.

    Static Methods
    --------------
    _transmittance(c_theta_i, s_theta_i, n_i, n_t)
        Calculates reflectance based on Fresnel equations.

    Parameters
    ----------
    n_medium : float
        Refractive index of the medium.
    """
    n_medium: float

    @staticmethod
    def _transmittance(c_theta_i, s_theta_i, n_i, n_t):
        """
        Adjusts transmittance due to boundary of LC system and background.

        Parameters
        ----------
        c_theta_i : float
            Cosine of the incident angle.
        s_theta_i : float
            Sine of the incident angle.
        n_i : float
            Refractive index of the incident medium.
        n_t : float
            Refractive index of the transmitting medium.

        Returns
        -------
        tuple
            Tuple containing:
                float : Transmittance for p-polarized light.
                float : Transmittance for s-polarized light.
        """
        n = n_i / n_t
        c_theta_t = np.sqrt(1 - (n * s_theta_i) ** 2)
        n_c_theta_t = n * c_theta_t
        n_c_theta_i = n * c_theta_i

        Rp = np.abs((n_c_theta_t - c_theta_i) / (n_c_theta_t + c_theta_i)) ** 2
        Rs = np.abs((n_c_theta_i - c_theta_t) / (n_c_theta_i + c_theta_t)) ** 2

        return 1 - Rp, 1 - Rs

    def transmittance(self, T, lc, no, ne, ind):
        """
        Calculates the adjusted transmittance based on Fresnel equations.

        Parameters
        ----------
        T : float
            Initial transmittance value.
        lc : object
            Object containing properties of the LC system.
        no : float
            Ordinary refractive index of the LC system.
        ne : float
            Extraordinary refractive index of the LC system.
        ind : int
            Index specifying the interface of the LC system.

        Returns
        -------
        float
            Adjusted transmittance value.
        """
        if not lc.interface[ind]:
            return T
        c_theta = lc.normal_z[ind]
        s_theta = np.sqrt(1 - c_theta**2)
        n1 = self.n_medium
        n2 = (2 * no + ne) / 3
        Tp, Ts = self._transmittance(c_theta, s_theta, n1, n2)
        return T * (Tp * c_theta + Ts * s_theta)


@dataclass
class EmpiricalDecay(TransmissionMode):
    """
    Represents empirical decay transmission mode for optics calculations.

    Attributes
    ----------
    theta0 : float, optional
        Threshold angle (default is 0.85).
    alpha : float, optional
        Decay coefficient (default is 0.06).

    Methods
    -------
    transmittance(T, lc, _no, _ne, ind)
        Calculates and returns the adjusted transmittance based on empirical decay.

    Parameters
    ----------
    theta0 : float, optional
        Threshold angle (default is 0.85).
    alpha : float, optional
        Decay coefficient (default is 0.06).
    """
    theta0: float = 0.85  # noqa: F841
    alpha: float = 0.06  # noqa: F841

    def transmittance(self, T, lc, _no, _ne, ind):
        """
        Calculates the adjusted transmittance based on empirical decay.

        Parameters
        ----------
        T : float
            Initial transmittance value.
        lc : object
            Object containing properties of the LC system.
        _no : float
            Unused parameter (ordinary refractive index of the LC system).
        _ne : float
            Unused parameter (extraordinary refractive index of the LC system).
        ind : int
            Index specifying the interface of the LC system.

        Returns
        -------
        float
            Adjusted transmittance value.
        """
        if not lc.interface[ind]:
            return T
        c_theta = lc.normal_z[ind]
        s_theta = np.sqrt(1 - c_theta**2)
        return T / (1 + np.exp((s_theta - self.theta0)) / self.alpha)
