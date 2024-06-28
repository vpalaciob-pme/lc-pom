import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import rotation

tau = 2 * np.pi

def abs2(c):
    """
    Compute the squared magnitude of a complex number.

    Parameters
    ----------
    c : complex
        The complex number.

    Returns
    -------
    float
        The squared magnitude.
    """
    return np.real(c) ** 2 + np.imag(c) ** 2

def normalize(v):
    """
    Normalize a vector to unit length.

    Parameters
    ----------
    v : np.ndarray
        The vector to normalize.

    Returns
    -------
    np.ndarray
        The normalized vector.
    """
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

def gaussian(x, mu, sigma):
    """
    Compute the value of a Gaussian function.

    Parameters
    ----------
    x : float
        The input value.
    mu : float
        The mean of the Gaussian.
    sigma : float
        The standard deviation of the Gaussian.

    Returns
    -------
    float
        The value of the Gaussian function at x.
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def normalized_gaussian(x, mu, sigma):
    """
    Compute the value of a normalized Gaussian function.

    Parameters
    ----------
    x : float
        The input value.
    mu : float
        The mean of the Gaussian.
    sigma : float
        The standard deviation of the Gaussian.

    Returns
    -------
    float
        The value of the normalized Gaussian function at x.
    """
    return gaussian(x, mu, sigma) / (sigma * np.sqrt(tau))

def piecewise_gaussian(x, mu, sigma1, sigma2):
    """
    Compute the value of a piecewise Gaussian function.

    Parameters
    ----------
    x : float
        The input value.
    mu : float
        The mean of the Gaussian.
    sigma1 : float
        The standard deviation of the Gaussian for values less than mu.
    sigma2 : float
        The standard deviation of the Gaussian for values greater than or equal to mu.

    Returns
    -------
    float
        The value of the piecewise Gaussian function at x.
    """
    sigma = sigma1 if x < mu else sigma2
    return gaussian(x, mu, sigma)

# These functions are copied from the mahotas package
def _convert(array, matrix, dtype=None):
    """
    Convert an array using a transformation matrix.

    Parameters
    ----------
    array : np.ndarray
        The input array.
    matrix : np.ndarray
        The transformation matrix.
    dtype : type, optional
        The desired data type of the output array, by default None.

    Returns
    -------
    np.ndarray
        The transformed array.
    """
    h, w, _ = array.shape
    array = array.transpose((2, 0, 1))
    array = array.reshape((3, h * w))
    array = np.dot(matrix, array)
    array = array.reshape((3, h, w))
    array = array.transpose((1, 2, 0))
    if dtype is not None:
        array = array.astype(dtype, copy=True)
    return array

def xyz2rgb(xyz, dtype=None):
    """
    Convert an image from XYZ color space to RGB color space.

    Parameters
    ----------
    xyz : np.ndarray
        The input image in XYZ color space.
    dtype : type, optional
        The desired data type of the output image, by default None.

    Returns
    -------
    np.ndarray
        The image in RGB color space.
    """
    transformation = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )

    return _convert(xyz, transformation, dtype)

def rgb2xyz(rgb, dtype=None):
    """
    Convert an image from RGB color space to XYZ color space.

    Parameters
    ----------
    rgb : np.ndarray
        The input image in RGB color space.
    dtype : type, optional
        The desired data type of the output image, by default None.

    Returns
    -------
    np.ndarray
        The image in XYZ color space.
    """
    transformation = np.array(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ]
    )

    return _convert(rgb, transformation, dtype)

def rotate(coords, directors, angles):
    """
    Rotate coordinates and directors by given Euler angles.

    Parameters
    ----------
    coords : np.ndarray
        The coordinates to rotate.
    directors : np.ndarray
        The directors to rotate.
    angles : array-like
        The Euler angles for rotation in degrees.

    Returns
    -------
    tuple
        The rotated coordinates and directors.
    """
    r = rotation.R.from_euler("xyz", angles, degrees=True).as_matrix()
    coords = np.matmul(r, coords.T).T
    directors = np.matmul(r, directors.T).T
    return coords, directors

def rotation_matrix(alpha: float):
    """
    Create a rotation matrix for a given angle around the z-axis.

    Parameters
    ----------
    alpha : float
        The angle of rotation in radians.

    Returns
    -------
    np.ndarray
        The 3x3 rotation matrix.
    """
    s_alpha = np.sin(alpha)
    c_alpha = np.cos(alpha)

    return np.array([[c_alpha, -s_alpha, 0], [s_alpha, c_alpha, 0], [0, 0, 1]])

def plot_image(intensity, vmax=None, savename=None):
    """
    Plot an intensity image.

    Parameters
    ----------
    intensity : np.ndarray
        The intensity image to plot.
    vmax : float, optional
        The maximum value for color scaling, by default None.
    savename : str, optional
        The filename to save the plot, by default None.
    """
    fig, ax = plt.subplots()
    if len(intensity.shape) == 3:
        image = np.transpose(intensity, [1, 0, 2])
    else:
        image = np.transpose(intensity)
    if vmax is None:
        vmax = np.max(image)
    im = ax.imshow(
        image,
        cmap=plt.get_cmap("bone"),
        interpolation="bicubic",
        origin="lower",
        vmax=vmax,
    )
    ax.set_ylim(0, image.shape[0] - 1)
    ax.set_xlim(0, image.shape[1] - 1)
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    ax.axis("off")
    plt.tight_layout(pad=0)
    dpi = matplotlib.rcParams["savefig.dpi"]
    fig.set_size_inches(5 * image.shape[1] / dpi, 5 * image.shape[0] / dpi)
    if savename is not None:
        plt.savefig(savename, pad_inches=0)
    return

def plot_hist(ys, savename=None):
    """
    Plot a histogram of intensity values.

    Parameters
    ----------
    ys : np.ndarray
        The intensity values.
    savename : str, optional
        The filename to save the plot, by default None.
    """
    _, ax = plt.subplots()
    ys = np.asarray(ys)

    upper = np.max(ys)
    if upper < 1.0e-2:
        upper = 1.0
    ax.hist(ys.flatten(), bins=np.linspace(0, upper, 51), density=True)
    ax.set_yscale("log")
    ax.set_xlabel("Intensity")
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    return