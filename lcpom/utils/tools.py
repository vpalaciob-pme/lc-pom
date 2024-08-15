import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import rotation


tau = 2 * np.pi


def abs2(c):
    return np.real(c) ** 2 + np.imag(c) ** 2


def normalize(v):
    """
    Normalize a vector by dividing each part by common number.

    After normalization the magnitude should be equal to ~1.
    """
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def normalized_gaussian(x, mu, sigma):
    return gaussian(x, mu, sigma) / (sigma * np.sqrt(tau))


def piecewise_gaussian(x, mu, sigma1, sigma2):
    sigma = sigma1 if x < mu else sigma2
    return gaussian(x, mu, sigma)


# These functions are copied from the mahotas package
def _convert(array, matrix, dtype=None):
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
    scikit-image
    http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_RGB.html
    """
    transformation = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )

    res = _convert(xyz, transformation, dtype)
    return res


def rgb2xyz(rgb, dtype=None):
    transformation = np.array(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ]
    )

    res = _convert(rgb, transformation, dtype)
    return res


def rotate(coords, directors, angles):
    r = rotation.R.from_euler("xyz", angles, degrees=True).as_matrix()
    coords = np.matmul(r, coords.T).T
    directors = np.matmul(r, directors.T).T
    return coords, directors


def rotation_matrix(alpha: float):
    """
    Matrix representation of a rotation operator.

    Arguments
    ---------

    alpha:
        Angle of rotation around rotation axis  [radians]

    Returns:
        3x3 matrix
    """

    s_alpha = np.sin(alpha)
    c_alpha = np.cos(alpha)

    return np.asarray([[c_alpha, -s_alpha, 0], [s_alpha, c_alpha, 0], [0, 0, 1]])


def plot_image(intensity, vmax=None, savename=None):
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
    # ax.set_title ("0$^o$")
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
    _, ax = plt.subplots()
    ys = np.asarray(ys)

    upper = np.max(ys)
    if upper < 1.0e-2:
        upper = 1.0
    image = ys
    # for image in ys:
    ax.hist(image.flatten(), bins=np.linspace(0, upper, 51), density=True)
    ax.set_yscale("log")
    ax.set_xlabel("Intensity")
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename)
    return
