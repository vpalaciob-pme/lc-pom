from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection

import numpy as np
from plum import dispatch, parametric
from scipy.interpolate import CubicSpline

from lcpom.optics import FullTransmission, TransmissionMode
from lcpom.utils.tools import normalized_gaussian


class Spectrum(ABC):
    """
    Base class for light spectra.
    """

    wavelengths: Collection[float]

    @abstractmethod
    def __init__(self, wavelengths):
        self.wavelengths = np.asarray(wavelengths)


@dataclass
class Monochromatic(Spectrum):
    """
    Spectrum for monochromatic light.
    """

    @dispatch
    def __init__(self, wavelength: float):
        super().__init__(np.array([wavelength]))

    @dispatch
    def __init__(self, wavelengths: Collection[float]):
        assert len(wavelengths) == 1, "For a multiple wavelengths use Polychromatic instead"
        super().__init__(wavelengths)


@dataclass
class Polychromatic(Spectrum):
    """
    Spectrum for polychromatic light.
    """

    @dispatch
    def __init__(self, wavelengths: Collection[float], skip_check: bool = False):
        assert len(wavelengths) > 1, "For a single wavelength use Monochromatic instead"
        super().__init__(wavelengths)


@dataclass
class LightSource:
    pass


@dataclass
class GaussianLEDLamp(LightSource):
    def intensity(self, wl):
        """
        Calculates the spectral radiance of the LED Lamp for a wavelength `wl` based on
        an approximate gaussian model.
        """
        return (
            0.15 * normalized_gaussian(wl, 0.45, 0.01)
            + 0.41 * normalized_gaussian(wl, 0.525, 0.05)
            + 0.37 * normalized_gaussian(wl, 0.625, 0.05)
            + 0.07 * normalized_gaussian(wl, 0.75, 0.05)
        )


@dataclass
class LEDLamp(LightSource):
    interpolator: CubicSpline

    def __init__(self, datafile, delimiter: str = ","):
        data = np.loadtxt(datafile, delimiter)
        self.interpolator = CubicSpline(data[:, 0], data[:, 1])

    def intensity(self, wl):
        """
        Calculates the spectral radiance of the LED Lamp for a wavelength `wl`.
        """
        return self.interpolator(wl)


DEFAULT_SPECTRUM = Polychromatic(np.arange(0.400, 0.681, 0.014))


@parametric
class IncidentLight:
    @classmethod
    def __infer_type_parameter__(cls, *args, **kwargs):
        return type(args[0]) if len(args) > 0 else type(DEFAULT_SPECTRUM)

    def __init__(
        self,
        spectrum: Spectrum = DEFAULT_SPECTRUM,
        alpha: float = 90.0,
        exposure: float = 1.0,
        source: LightSource = GaussianLEDLamp(),
        transmission_mode: TransmissionMode = FullTransmission(),
    ):
        """
        Characteristics of the incident light.

        Parameters
        ----------

        spectrum: Spectrum
            Discretized collection of wavelengths in the incident light.

        alpha:
            Polarizer angle in degrees.

        exposure:
            Exposure factor.

        source:
            Specifies the type of light source (uniform white light, LED lamp,
            or halogen lamp).

        reflection_transmission: Optional[Callable]
            Only valid for droplets. If provided, should compute the reflection
            attenuation.
        """

        self.spectrum = spectrum
        self.angle = alpha * np.pi / 180
        self.source = source
        self.exposure = exposure
        self.transmission_mode = transmission_mode
