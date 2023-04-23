from lcpom.pom.pomimage import POMFrame
import click


@click.command()
@click.option(
    "--mode",
    default="Single",
    type=click.Choice(["Single", "Multiple"], case_sensitive=False),
    show_default=True,
    help="Select LCPOM mode for calculating Intensity profiles. Valid options are: 1 Single wavelength, 2 Multiple wavelengths - full color spectrum",
)
@click.option(
    "--angle",
    default=0.0,
    type=float,
    show_default=True,
    help="Polarizer angle (in degrees)",
)
@click.option(
    "--exposure", default=1.0, type=float, show_default=True, help="Exposure factor"
)
@click.option(
    "--wls",
    type=(
        click.FloatRange(0.4, 0.68, clamp=True),
        click.FloatRange(0.4, 0.68, clamp=True),
        int,
    ),
    require=True,
    multiple=True,
    help='Wavelength interval in microns [wl0,wl1] in [wl2] steps. If "Single" mode is chosen, the calculation will only be done for wavelength=wl0.',
)
@click.option(
    "--path",
    require=True,
    type=click.Path(exists=True, file_okay=True, readable=True),
    help="Location of the order field file",
)
def run_lcpom(mode, angle, exposure, wls, path):
    """Simple program to run a LCPOM calculation.
    Inputs:
        - mode: 'Single' wavelength produces one intensity profile (BW)
                'Multiple' wavelengths calculates intensity profiles at each specified wavelength and reweights it to produce three images corresponding to RGB channels, as well as one full color POM
        - angle: Specifies the angle fo the lambda plate (?)
        - exposure
        - wls
        - path: Location of (vector or tensor) order field files
    Outputs:
        Intensity profiles
    """

    # Create the interval of wavelengths
    for wl in wls:
        wl0, wl1, nwl = wl
        lambdas = np.linspace(wl0, wl1, nwl, endpoint=True)

    modedict = {"Single": 1, "Multiple": 2}
    
    POMFrame(path, modedict, angle, exposure, lambdas)

    # POM_of_Frame(name.strip('\n'), mode = mode1, angle = angle1, exposureFactor = exposureFactor1, wl = wlStart, )
    

if __name__ == "__main__":
    run_lcpom()
