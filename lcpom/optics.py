from dataclasses import dataclass

import numpy as np


class TransmissionMode:
    pass


class FullTransmission(TransmissionMode):
    def transmittance(self, T, *_args):
        return T


@dataclass
class Fresnel(TransmissionMode):
    n_medium: float

    @staticmethod
    def _transmittance(c_theta_i, s_theta_i, n_i, n_t):
        """ "
        Adjusting transmittance due to boundary of LC system and background
        """
        n = n_i / n_t
        # `c_theta_t` may be complex and we may want to add 0j for this to work in general
        c_theta_t = np.sqrt(1 - (n * s_theta_i) ** 2)
        n_c_theta_t = n * c_theta_t
        n_c_theta_i = n * c_theta_i

        # Reflectance Fresnel equation for p-polarized light
        # Rₚ = |(n₁cos(θₜ) - n₂cos(θᵢ)) / (n₁cos(θₜ) + n₂cos(θᵢ))|²
        Rp = np.abs((n_c_theta_t - c_theta_i) / (n_c_theta_t + c_theta_i)) ** 2
        # Reflectance Fresnel equation for s-polarized light
        # Rₛ = |(n₁cos(θᵢ) - n₂cos(θₜ)) / (n₁cos(θᵢ) + n₂cos(θₜ))|²
        Rs = np.abs((n_c_theta_i - c_theta_t) / (n_c_theta_i + c_theta_t)) ** 2

        return 1 - Rp, 1 - Rs

    def transmittance(self, T, lc, no, ne, ind):
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
    theta0: float = 0.85  # noqa: F841
    alpha: float = 0.06  # noqa: F841

    def transmittance(self, T, lc, _no, _ne, ind):
        if not lc.interface[ind]:
            return T
        c_theta = lc.normal_z[ind]
        s_theta = np.sqrt(1 - c_theta**2)
        # Should the above be θ as oppossed to sin(θ)? That's,
        # theta = np.arccos(lc.normal_z[ind])
        return T / (1 + np.exp((s_theta - self.theta0)) / self.alpha)
