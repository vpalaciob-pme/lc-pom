from numbers import Number

import numpy as np
from PIL import Image, ImageOps
from plum import dispatch
from scipy.integrate import simpson

from lcpom.light import LightSource, IncidentLight
from lcpom.utils.compat import pairwise
from lcpom.utils.tools import piecewise_gaussian


@dispatch
def cie_xyz(source: LightSource, wl_min, wl_max, intervals: int = 20):
    """
    Computes CIE XYZ values by integrating the spectral radiance weighted by
    the color matching functions in the wavelength interval from `wl_min` to
    `wl_max`, with the wavelengths measured in micrometers.
    
    Parameters:
        source (LightSource):
            The light source object providing spectral radiance.
        wl_min (float):
            Minimum wavelength in micrometers for integration.
        wl_max (float):
            Maximum wavelength in micrometers for integration.
        intervals (int, optional):
            Number of intervals for wavelength sampling (default is 20).
    
    Returns:
        numpy.ndarray:
            CIE XYZ values computed by integrating the weighted spectral radiance.
    """
    wls = np.linspace(wl_min, wl_max, intervals)
    radiance = source.intensity(wls)
    xyzs = color_matching_xyz(1000 * wls)
    xyzs *= radiance
    return simpson(xyzs, wls, axis=0)


@dispatch
def cie_xyz(light: IncidentLight, intervals: int = 20):
    """
    Computes CIE XYZ values for an incident light spectrum by integrating the
    spectral radiance weighted by the color matching functions over each spectral
    interval defined in the incident light's spectrum.
    
    Parameters:
        light (IncidentLight):
            The incident light object containing source and spectrum information.
        intervals (int, optional):
            Number of intervals for wavelength sampling (default is 20).
    
    Returns:
        tuple:
            A tuple containing:
                numpy.ndarray:
                    Array of central wavelengths for each spectral interval.
                numpy.ndarray:
                    Array of CIE XYZ values corresponding to each spectral interval.
    """
    wls = []
    xyzs = []

    for wl_min, wl_max in pairwise(light.spectrum):
        wls.append((wl_min + wl_max) / 2)
        xyzs.append(cie_xyz(light.source, wl_min, wl_max, intervals))

    return np.asarray(wls), np.vstack(xyzs)

@dispatch
def color_matching_xyz(wl: Number):
    """
    Computes piecewise-Gaussian functions approximating the CIE XYZ
    color matching functions at a wavelength `wl` given in nanometers.
    
    Parameters:
        wl (Number):
            Wavelength in nanometers at which to compute the color matching functions.
    
    Returns:
        tuple:
            Tuple containing:
                float: x̅(λ) value.
                float: y̅(λ) value.
                float: z̅(λ) value.
    """
    # fmt: off
    x = (
        1.056 * piecewise_gaussian(wl, 599.8, 37.9, 31.0)
        + 0.362 * piecewise_gaussian(wl, 442.0, 16.0, 26.7)
        - 0.065 * piecewise_gaussian(wl, 501.1, 20.4, 26.2)
    )
    y = (
        0.821 * piecewise_gaussian(wl, 568.8, 46.9, 40.5)
        + 0.286 * piecewise_gaussian(wl, 530.9, 16.3, 31.1)
    )
    z = (
        1.217 * piecewise_gaussian(wl, 437.0, 11.8, 36.0)
        + 0.681 * piecewise_gaussian(wl, 459.0, 26.0, 13.8)
    )
    # fmt: on
    return (x, y, z)


@dispatch
def color_matching_xyz(wl: np.ndarray):  # noqa: F811 # pylint: disable=E0102
    """
    Computes piecewise-Gaussian functions approximating the CIE XYZ
    color matching functions for an array of wavelengths `wl` given
    in nanometers.
    
    Parameters:
        wl (np.ndarray):
            Array of wavelengths in nanometers at which to compute the color matching functions.
    
    Returns:
        numpy.ndarray:
            Array of CIE XYZ values corresponding to each wavelength in `wl`.
    """
    # Vectorize the scalar version of `color_matching_xyz`
    _color_matching_xyz = np.vectorize(color_matching_xyz.invoke(Number))

    xyz = np.asarray(_color_matching_xyz(wl)).T
    return xyz.squeeze()

@dispatch
def rgb_to_bw(image: np.ndarray):
    """
    Converts a RGB image represented as a NumPy array to grayscale using a simple median-based method.
    
    Parameters:
        image (np.ndarray):
            Input RGB image as a NumPy array.
    
    Returns:
        PIL.Image.Image:
            Grayscale image as a PIL Image.
    """
    if np.median(image) < 1:
        image = 255 * np.asarray(image, dtype=np.float32)
    return Image.fromarray(np.asarray(image, dtype=np.uint8))


@dispatch
def rgb_to_bw(path: str):
    """
    Converts a RGB image file to grayscale using PIL's ImageOps.

    Parameters:
        path (str):
            File path to the input RGB image.
    
    Returns:
        PIL.Image.Image:
            Grayscale image as a PIL Image.
    """
    image = Image.open(path)
    return ImageOps.grayscale(image)