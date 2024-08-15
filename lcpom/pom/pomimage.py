from dataclasses import dataclass
from itertools import product

import numpy as np
from plum import dispatch

from lcpom.light import IncidentLight, Monochromatic, Polychromatic
from lcpom.orderfield.lcsystems import LCGrid, refractive_indices
from lcpom.utils.tools import abs2, rotation_matrix, tau


@dataclass
class POMImage:
    """
    Class for the POM intensity profiles.
    """

    intensity: np.ndarray

    @dispatch
    def __init__(self, grid: LCGrid, light: IncidentLight[Monochromatic]):
        """
        Parameters
        ----------

        grid: LCGrid

        light: IncidentLight
        """
        # self.intensity = np.zeros((nx, ny))

    @dispatch
    def __init__(self, grid: LCGrid, light: IncidentLight[Polychromatic]):
        pass


@dataclass
class VoxelTransform:
    Theta: np.ndarray
    phi_0: float

    def __init__(self, delta, wl):
        self.Theta = np.zeros((2, 2), dtype=complex)
        self.phi_0 = tau * delta / wl  # 2π * Δ / λ₀

    def __call__(self, gamma: float, alpha: float, no, ne):
        """
        Jones matrix operator to calculate the phase retardation.

        Arguments
        ---------

        gamma:
            Angle of rotation around optical axis  [radians]

        alpha:
            Phase delay difference between ordinary and extraordinary axis

        no, ne:
            Ordinary and extraordinary refractive indices, respectively

        Returns
        -------

        2X2 matrix of the retardation operator
        """

        ne_gamma = np * ne / np.hypot(no * np.sin(gamma), ne * np.cos(gamma))

        So = np.exp(1j * self.phi_0 * no)  # S₁₁
        Se = np.exp(1j * self.phi_0 * ne_gamma)  # S₂₂

        ca = np.cos(alpha)
        sa = np.sin(alpha)

        self.Theta[0, 0] = So * ca
        self.Theta[0, 1] = So * -sa
        self.Theta[1, 0] = Se * sa
        self.Theta[1, 1] = Se * ca

        return self.Theta


@dispatch
def pom_image(lc: LCGrid, wl: float, Rp: np.ndarray, mode, update_ns, eps=1e-3):
    """
    Computes the Polarized Optical Microscopy (POM) intensity profile.

    Arguments
    ---------

    lc: LCGrid
        Liquid crystal system field information interpolated over a grid.

    wl: float
        Wavelength in nanometers.

    Rp: np.ndarray
        Polarizer rotation matrix.

    Notes
    -----

    It mainly follows the algorithm described in [1]_., but some details are drawn
    from reference [2]_.

    .. [1] Ellis PW, Pairam E, Fernández-Nieves A. "Simulating optical polarizing
        microscopy textures using Jones calculus: a review exemplified with nematic
        liquid crystal tori." J. Phys.D: Appl. Phys. 52(51) 213001 (2019).

    .. [2] Ondris‐Crawford R, et al. "Microscope textures of nematic droplets in
        polymer dispersed liquid crystals." J. Appl. Phys. 69(9) 6380-6 (1991).
    """
    # Discretization parameters
    Nx, Ny, Nz = lc.grid.shape

    intensity = np.zeros((Nx, Ny))
    # Incident polarization
    e_P = np.asarray([1, 0], dtype=complex)
    # Analyzer orientation
    e_A = np.asarray([0, 1], dtype=complex)
    # Function that returns each voxel's Jones transformation matrix
    voxel_transform = VoxelTransform(lc.grid.spacing[2], wl)

    ci = product(range(Nx), range(Ny))
    director = lc.director.reshape(-1, Nz, 3)
    no, ne = refractive_indices(wl, lc.material_params)

    for (i, j), nns in zip(ci, director):
        # Total Jones transformation matrix
        P = np.eye(2, dtype=complex)
        alpha_0 = 0.0
        T = 1.0

        for k, nn in enumerate(nns):
            if np.linalg.norm(nn) < eps:
                continue

            no, ne = update_ns(no, ne, lc.S[i, j, k])
            alpha_0, alpha, gamma = voxel_angles(Rp @ nn, alpha_0)
            Theta = voxel_transform(gamma, alpha, no, ne)
            # In-place alternative to P = Theta @ P
            np.matmul(Theta, P, out=P)

            # `mode.transmittance` checks internally if an interface is being crossed
            T = mode.transmittance(T, lc, no, ne, (i, j, k))

        # Exit voxel rotation matrix
        Rk = rotation_matrix(-alpha_0)
        # In-place alternative to P = Rk @ P
        np.matmul(Rk, P, out=P)

        intensity[i, j] = T * abs2(e_A @ (P @ e_P))

    return intensity


def voxel_angles(nn, alpha_0):
    gamma = np.arccos(np.abs(nn[2]))  # make the vector point along the positive Z axis
    alpha = np.arctan2(nn[1], nn[0])
    return alpha, alpha - alpha_0, gamma
