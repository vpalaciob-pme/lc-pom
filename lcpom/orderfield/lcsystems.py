from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.linalg import eig
from plum import dispatch
from scipy.interpolate import RBFInterpolator
from scipy.spatial.transform import Rotate

# Local imports
from lcpom.utils.tools import normalize


@dataclass
class GridInfo:
    length: np.ndarray
    shape: tuple
    size: int
    spacing: np.ndarray

    def __init__(self, length, shape, padding: float = 0.0):
        self.length = length * (1 + padding)
        self.shape = shape
        self.size = np.prod(shape)
        self.spacing = np.asarray([L / n for L, n in zip(length, shape)])


@dataclass
class Grid:
    info: GridInfo
    centers: Optional[np.ndarray]

    def __init__(self, length, shape, lazy=True):
        self.info = GridInfo(length, shape)
        self.centers = None if lazy else make_grid(self.info)

    def __getitem__(self, indices):
        return getindex(self.info, self.centers, indices)


@dispatch
def make_grid(info: GridInfo, dtype=np.float32):
    lx, ly, lz = info.length / 2
    nx, ny, nz = info.shape

    x = np.linspace(-lx, lx, nx)
    y = np.linspace(-ly, ly, ny)
    z = np.linspace(-lz, lz, nz)
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")

    centers = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1, dtype=dtype)
    return centers.reshape(nx, ny, nz, 3)


@dispatch
def getindex(info: GridInfo, centers: type(None), *indices):
    lx, ly, lz = info.length
    dx, dy, dz = info.spacing
    i, j, k = flatten(*indices)
    return np.array([-lx / 2 + i * dx, -ly / 2 + j * dy, -lz / 2 + k * dz])


@dispatch
def getindex(info: GridInfo, centers, *indices):
    return centers.__getitem__(*indices)


@dispatch
def flatten(*indices):
    return indices


@dispatch
def flatten(indices: tuple):
    return indices


# Is this needed anywhere?
class QTensor:
    v: np.ndarray

    def __init__(self, S: float, n: np.ndarray):
        Q0 = S * (n[0] * n[0] - 1.0 / 3.0)
        Q1 = S * n[0] * n[1]
        Q2 = S * n[0] * n[2]
        Q3 = S * (n[1] * n[1] - 1.0 / 3.0)
        Q4 = S * n[1] * n[2]
        self.v = np.array([Q0, Q1, Q2, Q3, Q4])

    def as_matrix(self):
        """
        Returns the tensor information as a symmetric and traceless 3x3 matrix.
        Useful when converting the independent entries of Q onto the tensor form.
        """
        v = self.v
        return np.array(
            [
                [v[0], v[1], v[2]],
                [v[1], v[3], v[4]],
                [v[2], v[4], -v[0] - v[3]],
            ]
        )


# Is this needed anywhere?
def order_parameter(tensor: QTensor):
    Q = tensor.as_matrix()
    vals, vecs = eig(Q)
    i = vals.argmax()
    S = 1.5 * vals[i]
    n = vecs[i]
    return S, n


class MaterialParams:
    pass


@dataclass
class ThreeBandModelParams(MaterialParams):
    l1: float
    l2: float
    n0e: float
    n0o: float
    g1e: float
    g2e: float
    g1o: float
    g2o: float


# These parameters could be read from a dictionary depending on the material input.
PARAMS_5CB = ThreeBandModelParams(
    l1=0.210,
    l2=0.282,
    n0e=0.455,
    n0o=0.414,
    g1e=2.325,
    g2e=1.397,
    g1o=1.352,
    g2o=0.470,
)


class LCGrid:
    @dispatch
    def __init__(
        self,
        grid: Grid,
        order_parameter,
        director,
        interface,
        normal_z,
        material_params: MaterialParams = PARAMS_5CB,
    ):
        """
        LCGrid is a class that handles the LC information once scalar and director order
        fields are interpolated onto grid
        """
        self.grid = grid
        self.S = order_parameter
        self.director = director
        self.interface = interface
        self.normal_z = normal_z
        self.material_params = material_params


class RefractiveIndicesUpdater:
    pass


class OrderParameterIndependant(RefractiveIndicesUpdater):
    @staticmethod
    def update_ns(no, ne, S):
        return no, ne


class OrderParameterDependant(RefractiveIndicesUpdater):
    @staticmethod
    def update_ns(no, ne, S, S0: float = 0.57):
        delta_n = (ne - no) / S0
        n_mean = (ne + 2 * no) / 3
        ne = n_mean + 2 / 3 * S * delta_n
        no = n_mean - 1 / 3 * S * delta_n
        return no, ne


@dispatch
def refractive_indices(wl: float, p: ThreeBandModelParams):
    """
    Calculates the refractive indices (n_o, n_e) with the three-band model
    parameters `p` at the wavelength `wl` and order parameter `S`.
    """
    lambda_sq = wl**2
    lambda1_sq = p.l1**2
    lambda2_sq = p.l2**2

    r1 = (lambda_sq * lambda1_sq) / (lambda_sq - lambda1_sq)
    r2 = (lambda_sq * lambda2_sq) / (lambda_sq - lambda2_sq)

    ne = 1 + p.n0e + p.g1e * r1 + p.g2e * r2
    no = 1 + p.n0o + p.g1o * r1 + p.g2o * r2

    return no, ne


def interpolate(system: LCGrid, delta: float = 0.1, method="thin_plate_spline"):
    """
    Interp_frame interpolates the order field data onto a grid with finer resolution
    """

    # Make grid according to size of the ellipsoid
    grid = make_grid(system.L, delta=delta)

    # Interpolate data onto finer grid
    centroid = np.mean(system.coords, axis=1)
    idx = np.where(
        ellip1(grid.xyz.T, system.L_box, centroid) > 0
    )  # Finds nodes where LC does not exist

    interpolate_director = RBFInterpolator(
        system.centers, system.director, kernel=method, smoothing=0.1, neighbors=12
    )
    nn = interpolate_director(grid.xyz)
    nn = normalize(nn)
    nn[idx] = 0

    interpolate_order_paramter = RBFInterpolator(
        system.centers, system.S, kernel=method, smoothing=0.1, neighbors=12
    )
    ss = interpolate_order_paramter(grid.xyz)
    ss[idx] = 0

    # Save the interpolated director field from directory2
    # if ( system.Sorder.any() != None):
    #    ss = np.reshape(ss, (len(ss),1))
    #    write_txt_s(rr, nn,ss, consts0, l_box, info, directory2)
    # else:
    #    write_txt(rr, nn, consts0, l_box,info,directory2)

    return LCGrid(grid, ss, nn, PARAMS_5CB)


def ellip1(r, L, center):
    x = (r[0] - center[0]) * 2 / L[0]
    y = (r[1] - center[1]) * 2 / L[1]
    z = (r[2] - center[2]) * 2 / L[2]
    return x**2 + y**2 + z**2 - 1


def write_orig(rr, nn, ss, info, euler_angles, directory1):
    ss = ss.reshape([len(ss), 1])
    if ss.any() is None:
        X0 = np.hstack([rr, nn, ss])
    else:
        X0 = np.hstack([rr, nn])
    top = "# Euler angles: %.2f, %.2f, %.2f" % (
        euler_angles[0],
        euler_angles[1],
        euler_angles[2],
    )
    fname = directory1 + info + "-original-directors.txt"
    np.savetxt(fname, X0, fmt="%.4f", delimiter="\t", header=top)
    return


def write_txt(rr, nn, consts0, l_box, info, directory2):
    [nx, ny, nz, dx, dy, dz] = consts0
    X0 = np.hstack([rr, nn])
    header = "Interpolated director file\n"
    info0 = section_info("Grid Info:", "Nx Ny Nz dx dy dz")
    info1 = section_info("Ranges:", "x_min x_max y_min y_max z_min z_max")
    info2 = section_info("Data:", "x y z n_x n_y n_z")
    line0 = np.asarray([nx, ny, nz, dx, dy, dz])
    line1 = np.asarray(
        [
            -l_box[0] / 2,
            l_box[0] / 2,
            -l_box[1] / 2,
            l_box[1] / 2,
            -l_box[2] / 2,
            l_box[2] / 2,
        ]
    )

    X = np.vstack([line0, line1, X0])
    top = header + info0 + info1 + info2
    fname = directory2 + info + "-interpolated-directors.txt"
    np.savetxt(fname, X, fmt="%.4f", delimiter="\t", header=top)
    return


def section_info(title, cols, sep="\t", end="\n"):
    if len(title) != 0:
        title += "\n"
    return title + sep.join(cols.split()) + end


def write_txt_s(rr, nn, ss, consts0, l_box, info, directory2):
    [nx, ny, nz, dx, dy, dz] = consts0
    X0 = np.hstack([rr, nn, ss])
    header = "Interpolated director file\n"
    info0 = section_info("Grid Info:", "Nx Ny Nz dx dy dz S_max")
    info1 = section_info("Ranges:", "x_min x_max y_min y_max z_min z_max S_mean")
    info2 = section_info("Data:", "x y z n_x n_y n_z S")
    line0 = np.asarray([nx, ny, nz, dx, dy, dz, np.max(ss)])
    line1 = np.asarray(
        [
            -l_box[0] / 2,
            l_box[0] / 2,
            -l_box[1] / 2,
            l_box[1] / 2,
            -l_box[2] / 2,
            l_box[2] / 2,
            np.mean(ss[np.where(ss > 0)]),
        ]
    )

    X = np.vstack([line0, line1, X0])
    top = header + info0 + info1 + info2
    fname = directory2 + info + "-interpolated-directors.txt"
    np.savetxt(fname, X, fmt="%.4f", delimiter="\t", header=top)
    return


def read_rotate(fname, scaling=1.0, euler_angles=np.asarray([0, 0, 0])):
    # Read original director field from directory1

    X = np.loadtxt(fname, dtype=np.float32)
    if X.shape[1] == 7:
        print("Has S")
        ss0 = X[:, 6]
        print(ss0.shape)
    else:
        ss0 = np.asarray([None])

    coords = X[:, :3] * scaling
    directors = X[:, 3:6]

    print("Coords shape", coords.shape)
    print("Directors shape", directors.shape)

    # Find ellipsoid size
    # L, centroid = find_L(coords)
    L = np.asarray(
        [
            max(coords[:, 0]) - min(coords[:, 0]),
            max(coords[:, 1]) - min(coords[:, 1]),
            max(coords[:, 2]) - min(coords[:, 2]),
        ]
    )
    centroid = np.asarray(
        [np.mean(coords[:, 0]), np.mean(coords[:, 1]), np.mean(coords[:, 2])]
    )
    print("L:", L, "Centroid:", centroid)

    # R ecenter
    coords[:, 0] -= centroid[0]
    coords[:, 1] -= centroid[1]
    coords[:, 2] -= centroid[2]
    print("Recentered")

    # Rotate
    if euler_angles.any() != 0:
        print(
            "Rotation around x, y, z in order by %.2f, %.2f, %.2f degrees"
            % (euler_angles[0], euler_angles[1], euler_angles[2])
        )
        coords, directors = Rotate(coords, directors, euler_angles)

    # Correct signs
    print("Correct signs of original director")
    signs = np.power(-1, (np.sum(np.multiply(directors, coords), 1) > 0))
    # signs =np.power (-1,directors[:,0]<0)
    directors[:, 0] = directors[:, 0] * signs
    directors[:, 1] = directors[:, 1] * signs
    directors[:, 2] = directors[:, 2] * signs

    return coords, directors, ss0, L


def plot_from_existed(
    fname_orig, fname_interp, info, scaling=1.0, euler_angles=np.asarray([0, 0, 0])
):
    # the original data
    coords, directors, ss0, L = read_rotate(
        fname_orig, scaling=scaling, euler_angles=euler_angles
    )

    # Load the interpolated field
    X = np.loadtxt(fname_interp, dtype=np.float32)
    # rr = X[2:, :3]
    # nn = X[2:, 3:6]
    [Nx, Ny, Nz] = np.asarray(X[0, :3], dtype=np.int32)
    [dx, dy, dz] = X[0, 3:6]
    [x_min, x_max, y_min, y_max, z_min, z_max] = X[1, 0:6]
    # if X.shape[1] == 7:
    #     print("Has S")
    #     ss = X[2:, 6]
    # else:
    #     ss = np.asarray([None])
    print("Number of data points:", Nx * Ny * Nz)
    print("dx = %.2f" % (dx))

    # plot and save
    # plot_final(rr, nn, ss, coords, directors, L, info, directory2)

    return
