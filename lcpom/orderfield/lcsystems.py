from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.linalg import eig
from plum import dispatch
from scipy.interpolate import RBFInterpolator
from scipy.spatial.transform import Rotation

# Local imports
from lcpom.utils.tools import normalize



@dataclass
class GridInfo:
    """
    Stores information about a grid.

    Attributes
    ----------
    length : np.ndarray
        Dimensions of the box that encloses the LC system (Lx,Ly,Lz)
    shape : tuple
        Shape of the coordinates array. (nx, ny, nz)
    size : int
        Total number of points in the grid.
    spacing : np.ndarray
        Spacing between grid points in each dimension.
    upper : np.ndarray
        Upper bounds of the grid.

    Parameters
    ----------
    length : np.ndarray
        Dimensions of the box that encloses the LC system (Lx,Ly,Lz)
    shape : tuple
        Shape of the coordinates array. (nx, ny, nz)
    padding : float, optional
        Padding factor applied to the length of the grid (default is 0.0).
    """

    length: np.ndarray
    shape: tuple
    size: int
    spacing: np.ndarray
    upper: np.ndarray

    def __init__(self, length, shape, padding: float = 0.0):
        self.length = np.asarray(length) * (1 + padding)
        self.shape = shape
        self.size = np.prod(shape)
        self.spacing = np.asarray([L / n for L, n in zip(length, shape)])
        self.upper = (self.length - self.spacing) / 2


@dataclass
class Grid:
    """
    Represents a grid with associated grid information.

    Attributes
    ----------
    info : GridInfo
        Information about the grid.
    centers : Optional[np.ndarray]
        Centers of the grid points, if precomputed.

    Parameters
    ----------
    length : np.ndarray
        Dimensions of the box that encloses the LC system (Lx,Ly,Lz)
    shape : tuple
        Shape of the coordinates array. (nx, ny, nz)
    lazy : bool, optional
        If True, centers are not precomputed (default is True).
    """

    info: GridInfo
    centers: Optional[np.ndarray]

    def __init__(self, length, shape, lazy=True):
        self.info = GridInfo(length, shape)
        self.centers = None if lazy else make_grid(self.info)

    def __getitem__(self, indices):
        """
        Retrieve the grid point at the specified indices.

        Parameters
        ----------
        indices : tuple
            Indices to access the grid point.

        Returns
        -------
        np.ndarray
            Coordinates of the grid point at the specified indices.
        """
        return getindex(self.info, self.centers, indices)


@dispatch
def make_grid(info: GridInfo, dtype=np.float32):
    """
    Create a grid of points based on the provided grid information.

    Parameters
    ----------
    info : GridInfo
        Information about the grid.
    dtype : type, optional
        Data type of the grid points (default is np.float32).

    Returns
    -------
    np.ndarray
        Grid points arranged in a 3D array.
    """

    lx, ly, lz = info.upper
    nx, ny, nz = info.shape

    x = np.linspace(-lx, lx, nx)
    y = np.linspace(-ly, ly, ny)
    z = np.linspace(-lz, lz, nz)
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")

    centers = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1, dtype=dtype)
    return centers.reshape(nx, ny, nz, 3)


@dispatch
def getindex(info: GridInfo, centers: type(None), *indices):
    """
    Retrieve the coordinates of a grid point specified by indices.

    Parameters
    ----------
    info : GridInfo
        Information about the grid.
    centers : type(None)
        Centers of the grid points (unused parameter).
    indices : tuple
        Indices to access the grid point.

    Returns
    -------
    np.ndarray
        Coordinates of the grid point at the specified indices.

    Raises
    ------
    IndexError
        If indices are out of bounds.
    """

    lx, ly, lz = info.upper
    dx, dy, dz = info.spacing
    nx, ny, nz = info.shape
    i, j, k = flatten(*indices)
    if not (-nx <= i < nx and -ny <= j < ny and -nz <= k < nz):
        raise IndexError(f"indices {i, j, k} out of bounds")
    return np.array([-lx + (i % nx) * dx, -ly + (j % ny) * dy, -lz + (k % nz) * dz])


@dispatch
def getindex(info: GridInfo, centers, *indices):
    """
    Retrieve the coordinates of a grid point specified by indices using precomputed centers.

    Parameters
    ----------
    info : GridInfo
        Information about the grid.
    centers : np.ndarray
        Centers of the grid points.
    indices : tuple
        Indices to access the grid point.

    Returns
    -------
    np.ndarray
        Coordinates of the grid point at the specified indices.
    """

    return centers.__getitem__(*indices)


@dispatch
def flatten(indices: tuple):
    """
    Flatten indices into separate components.

    Parameters
    ----------
    indices : tuple
        Indices to flatten.

    Returns
    -------
    tuple
        Flattened indices.
    """

    return indices


@dispatch
def flatten(*indices):
    """
    Flatten multiple indices into separate components.

    Parameters
    ----------
    indices : tuple
        Indices to flatten.

    Returns
    -------
    tuple
        Flattened indices.
    """

    return indices

# Is this needed anywhere? -> Yes! For when we define ansatz and other input forms
class QTensor:
    """
    Represents a Q-tensor in a liquid crystal system.

    Attributes
    ----------
    v : np.ndarray
        Array representing the Q-tensor elements.

    Parameters
    ----------
    S : float
        Order parameter of the liquid crystal.
    n : np.ndarray
        Director vector of the liquid crystal.
    """

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
        Returns the Q-tensor as a symmetric and traceless 3x3 matrix.

        Returns
        -------
        np.ndarray
            Symmetric and traceless 3x3 matrix representation of the Q-tensor.
        """
        v = self.v
        return np.array(
            [
                [v[0], v[1], v[2]],
                [v[1], v[3], v[4]],
                [v[2], v[4], -v[0] - v[3]],
            ]
        )


def order_parameter(tensor: QTensor):
    """
    Calculates the order parameter S and director vector n from a Q-tensor.

    Parameters
    ----------
    tensor : QTensor
        Q-tensor of the liquid crystal.

    Returns
    -------
    tuple
        Tuple containing:
            float : Order parameter S.
            np.ndarray : Director vector n.
    """
    Q = tensor.as_matrix()
    vals, vecs = eig(Q)
    i = vals.argmax()
    S = 1.5 * vals[i]
    n = vecs[i]
    return S, n


class MaterialParams:
    """
    Base class for material parameters in liquid crystal models.
    """
    pass


@dataclass
class ThreeBandModelParams(MaterialParams):
    """
    Material parameters for the three-band model in liquid crystal systems.

    Attributes
    ----------
    l1 : float
        Wavelength parameter 1.
    l2 : float
        Wavelength parameter 2.
    n0e : float
        Extraordinary refractive index offset.
    n0o : float
        Ordinary refractive index offset.
    g1e : float
        Extraordinary refractive index coefficient 1.
    g2e : float
        Extraordinary refractive index coefficient 2.
    g1o : float
        Ordinary refractive index coefficient 1.
    g2o : float
        Ordinary refractive index coefficient 2.
    """

    l1: float
    l2: float
    n0e: float
    n0o: float
    g1e: float
    g2e: float
    g1o: float
    g2o: float


# Example parameters for the 5CB liquid crystal material
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


@dataclass
class LCGrid:
    """
    Represents a grid with interpolated LC (liquid crystal) information.

    Parameters
    ----------
    grid : Grid
        Grid object storing information about the spatial grid.
    order_parameter : np.ndarray
        Array of order parameter values. Size of the array should match those of grid: grid.info.shape 
    director : np.ndarray
        Array of director vector components. Size of the array: (grid.info.size,3) 
    interface : np.ndarray
        Array indicating interfaces within the grid. 1 for nodes representing LC and 0 for the rest
    normal_z : np.ndarray
        Array of normal vectors to interfaces.
    material_params : MaterialParams, optional
        Parameters describing the material properties (default is PARAMS_5CB).
    """

    grid: Grid
    order_parameter: np.ndarray
    director: np.ndarray
    interface: np.ndarray
    normal_z: np.ndarray
    material_params: MaterialParams = field(default_factory=lambda: PARAMS_5CB)


class RefractiveIndicesUpdater:
    """
    Base class for updating refractive indices of liquid crystal systems.
    """
    pass


class OrderParameterIndependant(RefractiveIndicesUpdater):
    """
    Updates refractive indices independently of the order parameter S.

    Methods
    -------
    update_ns(no, ne, S)
        Updates and returns refractive indices no and ne.

    Parameters
    ----------
    no : float
        Initial ordinary refractive index.
    ne : float
        Initial extraordinary refractive index.
    S : float
        Order parameter of the liquid crystal.
    """

    @staticmethod
    def update_ns(no, ne, S):
        """
        Updates refractive indices no and ne independently of the order parameter S.

        Parameters
        ----------
        no : float
            Initial ordinary refractive index.
        ne : float
            Initial extraordinary refractive index.
        S : float
            Order parameter of the liquid crystal.

        Returns
        -------
        tuple
            Tuple containing:
                float : Updated ordinary refractive index.
                float : Updated extraordinary refractive index.
        """
        return no, ne


class OrderParameterDependant(RefractiveIndicesUpdater):
    """
    Updates refractive indices dependent on the order parameter S.

    Methods
    -------
    update_ns(no, ne, S, S0: float = 0.57)
        Updates and returns refractive indices no and ne.

    Parameters
    ----------
    no : float
        Initial ordinary refractive index.
    ne : float
        Initial extraordinary refractive index.
    S : float
        Order parameter of the liquid crystal.
    S0 : float, optional
        Reference order parameter (default is 0.57).
    """

    @staticmethod
    def update_ns(no, ne, S, S0: float = 0.57):
        """
        Updates refractive indices no and ne dependent on the order parameter S.

        Parameters
        ----------
        no : float
            Initial ordinary refractive index.
        ne : float
            Initial extraordinary refractive index.
        S : float
            Order parameter of the liquid crystal.
        S0 : float, optional
            Reference order parameter (default is 0.57).

        Returns
        -------
        tuple
            Tuple containing:
                float : Updated ordinary refractive index.
                float : Updated extraordinary refractive index.
        """
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

    Parameters
    ----------
    wl : float
        Wavelength in micrometers.
    p : ThreeBandModelParams
        Parameters for the three-band model.

    Returns
    -------
    tuple
        Tuple containing:
            float : Ordinary refractive index (n_o).
            float : Extraordinary refractive index (n_e).
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
    Interpolates the order field data onto a grid with finer resolution.

    Parameters
    ----------
    system : LCGrid
        The liquid crystal grid system containing the data to be interpolated.
    delta : float, optional
        The grid spacing for the interpolation, by default 0.1.
    method : str, optional
        The interpolation method to use, by default "thin_plate_spline".

    Returns
    -------
    LCGrid
        The interpolated liquid crystal grid.
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
    """
    Calculates the ellipsoid equation for given points.

    Parameters
    ----------
    r : array_like
        Coordinates of the points.
    L : array_like
        Lengths of the ellipsoid along each axis.
    center : array_like
        Center of the ellipsoid.

    Returns
    -------
    ndarray
        Values of the ellipsoid equation for the given points.
    """

    x = (r[0] - center[0]) * 2 / L[0]
    y = (r[1] - center[1]) * 2 / L[1]
    z = (r[2] - center[2]) * 2 / L[2]
    return x**2 + y**2 + z**2 - 1


def write_orig(rr, nn, ss, info, euler_angles, directory1):
    """
    Writes the original director field data to a file.

    Parameters
    ----------
    rr : array_like
        Coordinates of the points.
    nn : array_like
        Director vectors at the points.
    ss : array_like
        Scalar order parameters at the points.
    info : str
        Additional information to include in the file header.
    euler_angles : array_like
        Euler angles used for rotation.
    directory1 : str
        Directory where the file will be saved.
    """

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
    """
    Writes the interpolated director field data to a file.

    Parameters
    ----------
    rr : array_like
        Coordinates of the points.
    nn : array_like
        Director vectors at the points.
    consts0 : array_like
        Grid constants including dimensions and spacings.
    l_box : array_like
        Dimensions of the simulation box.
    info : str
        Additional information to include in the file header.
    directory2 : str
        Directory where the file will be saved.
    """

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
    """
    Formats section information for the file header.

    Parameters
    ----------
    title : str
        Title of the section.
    cols : str
        Column names.
    sep : str, optional
        Separator for the columns, by default "\t".
    end : str, optional
        End of line character, by default "\n".

    Returns
    -------
    str
        Formatted section information.
    """

    if len(title) != 0:
        title += "\n"
    return title + sep.join(cols.split()) + end


def write_txt_s(rr, nn, ss, consts0, l_box, info, directory2):
    """
    Writes the interpolated director field data and scalar order parameter to a file.

    Parameters
    ----------
    rr : array_like
        Coordinates of the points.
    nn : array_like
        Director vectors at the points.
    ss : array_like
        Scalar order parameters at the points.
    consts0 : array_like
        Grid constants including dimensions and spacings.
    l_box : array_like
        Dimensions of the simulation box.
    info : str
        Additional information to include in the file header.
    directory2 : str
        Directory where the file will be saved.
    """

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


def rotate(coords, directors, angles):
    """
    Rotates the coordinates and directors by the given Euler angles.

    Parameters
    ----------
    coords : array_like
        Coordinates of the points.
    directors : array_like
        Director vectors at the points.
    angles : array_like
        Euler angles for the rotation.

    Returns
    -------
    tuple
        Rotated coordinates and director vectors.
    """

    R = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    coords = (R @ coords.T).T
    directors = (R @ directors.T).T
    return coords, directors


def read_rotate(fname, scaling=1.0, euler_angles=np.asarray([0, 0, 0])):
    """
    Reads and rotates the director field data from a file.

    Parameters
    ----------
    fname : str
        Filename of the input file.
    scaling : float, optional
        Scaling factor for the coordinates, by default 1.0.
    euler_angles : array_like, optional
        Euler angles for the rotation, by default np.asarray([0, 0, 0]).

    Returns
    -------
    tuple
        Rotated coordinates, director vectors, scalar order parameters, and ellipsoid dimensions.
    """

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
        coords, directors = rotate(coords, directors, euler_angles)

    # Correct signs
    print("Correct signs of original director")
    signs = np.power(-1, (np.sum(np.multiply(directors, coords), 1) > 0))
    # signs =np.power (-1,directors[:,0]<0)
    directors[:, 0] = directors[:, 0] * signs
    directors[:, 1] = directors[:, 1] * signs
    directors[:, 2] = directors[:, 2] * signs

    return coords, directors, ss0, L

def plot_from_existed(fname_orig, fname_interp, info, scaling=1.0, euler_angles=np.asarray([0, 0, 0])):
    """
    Plots the original and interpolated director fields.

    Parameters
    ----------
    fname_orig : str
        Filename of the original director field data.
    fname_interp : str
        Filename of the interpolated director field data.
    info : str
        Additional information for the plot.
    scaling : float, optional
        Scaling factor for the coordinates, by default 1.0.
    euler_angles : array_like, optional
        Euler angles for the rotation, by default np.asarray([0, 0, 0]).
    """
    
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
