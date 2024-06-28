from dataclasses import dataclass
from typing import Collection

import numpy as np
from plum import dispatch
from scipy.interpolate import CubicSpline

from lcpom.optics import FullTransmission, TransmissionMode
from lcpom.utils.tools import normalized_gaussian


@dataclass
class Spectrum:
    """
    Base class for light spectra.
    """

    wavelengths: Collection[float]

    @dispatch
    def __init__(self, wavelengths: Collection[float], skip_check: bool = False):
        """
        Initializes a Spectrum object with given wavelengths.

        Parameters:
            wavelengths (Collection[float]):
                Collection of wavelengths for the spectrum.
            skip_check (bool, optional):
                Flag to skip length check for wavelengths (default is False).
                If False, ensures more than one wavelength is provided.

        Raises:
            AssertionError: If skip_check is False and only one wavelength is provided.
        """
        if not skip_check:
            assert (
                len(wavelengths) > 1
            ), "For a single wavelength use Monochrome instead"
        self.wavelengths = wavelengths


@dataclass
class Monochrome(Spectrum):
    """
    Special case of Spectrum for monochromatic light.
    """

    @dispatch
    def __init__(self, wavelength: float):
        """
        Initializes a Monochrome object with a single wavelength.

        Parameters:
            wavelength (float):
                Wavelength for the monochromatic light.
        """
        super().__init__([wavelength], skip_check=True)


@dataclass
class LightSource:
    pass


@dataclass
class GaussianLEDLamp(LightSource):
    """
    Represents a Gaussian model LED lamp as a light source.
    """

    def intensity(self, wl):
        """
        Calculates the spectral radiance of the LED Lamp for a wavelength `wl` based on
        an approximate Gaussian model.

        Parameters:
            wl (float):
                Wavelength at which to calculate the spectral radiance.

        Returns:
            float:
                Spectral radiance value.
        """
        return (
            0.15 * normalized_gaussian(wl, 0.45, 0.01)
            + 0.41 * normalized_gaussian(wl, 0.525, 0.05)
            + 0.37 * normalized_gaussian(wl, 0.625, 0.05)
            + 0.07 * normalized_gaussian(wl, 0.75, 0.05)
        )


@dataclass
class LEDLamp(LightSource):
    """
    Represents an LED lamp as a light source using interpolation for spectral radiance.
    """

    interpolator: CubicSpline

    def __init__(self, datafile, delimiter: str = ","):
        """
        Initializes an LEDLamp object with spectral data from a file.

        Parameters:
            datafile (str):
                Path to the file containing wavelength-intensity data.
            delimiter (str, optional):
                Delimiter used in the datafile (default is ",").
        """
        data = np.loadtxt(datafile, delimiter)
        self.interpolator = CubicSpline(data[:, 0], data[:, 1])

    def intensity(self, wl):
        """
        Calculates the spectral radiance of the LED Lamp for a wavelength `wl`.

        Parameters:
            wl (float):
                Wavelength at which to calculate the spectral radiance.

        Returns:
            float:
                Spectral radiance value.
        """
        return self.interpolator(wl)

class IncidentLight:
    def __init__(
        self,
        spectrum: Spectrum = Spectrum(np.arange(0.400, 0.681, 0.014)),
        alpha: float = 90.0,
        exposure: float = 1.0,
        source: LightSource = GaussianLEDLamp(),
        transmission_mode: TransmissionMode = FullTransmission(),
    ):
        """
        Characteristics of the incident light.

        Parameters
        ----------
        spectrum : Spectrum
            Discretized collection of wavelengths in the incident light.
        
        alpha : float
            Polarizer angle in degrees.
        
        exposure : float
            Exposure factor.
        
        source : LightSource
            Specifies the type of light source (uniform white light, LED lamp,
            or halogen lamp).
        
        transmission_mode : TransmissionMode
            Transmission mode of the incident light.
        """
        self.spectrum = spectrum
        self.angle = alpha * np.pi / 180
        self.source = source
        self.exposure = exposure
        self.transmission_mode = transmission_mode