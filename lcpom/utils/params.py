import numpy as np
import math

"""
We are adding two convenient constants to transform angles for trigonometric functions
As well as a few parameters that are the default to perform a POM calculation
"""

deg2rad = 180.0/math.pi
rad2deg = math.pi/180.0

angle = 0
case = 2
num = 3
wl = np.arange (0.4, 0.68, 0.014)
exposureFactor = 1.5

#
#   Refractive index
#  The refractive indices depend on wavelengths (and temperature).
#
#  Reference:
#
#  WU et al. Optical Engineering 1993 32(8) 1775
#  Li et al. Journal of Applied Physics 96, 19 (2004)


def calc_n(lamb):

    l1 = 0.210; l2 = 0.282;
    n0e = 0.455; g1e = 2.325; g2e = 1.397
    n0o = 0.414; g1o = 1.352; g2o = 0.470

    n_e = 1 + n0e + g1e*(lamb**2 * l1**2)/(lamb**2-l1**2) + g2e*(lamb**2 * l2**2)/(lamb**2-l2**2)
    n_o = 1 + n0o + g1o*(lamb**2 * l1**2)/(lamb**2-l1**2) + g2o*(lamb**2 * l2**2)/(lamb**2-l2**2)

    return n_o, n_e

def calc_n_s(lamb,s):

    l1 = 0.210; l2 = 0.282;
    n0e = 0.455; g1e = 2.325; g2e = 1.397
    n0o = 0.414; g1o = 1.352; g2o = 0.470

    n_e = 1 + n0e + g1e*(lamb**2 * l1**2)/(lamb**2-l1**2) + g2e*(lamb**2 * l2**2)/(lamb**2-l2**2)
    n_o = 1 + n0o + g1o*(lamb**2 * l1**2)/(lamb**2-l1**2) + g2o*(lamb**2 * l2**2)/(lamb**2-l2**2)

    S0 = 0.68
    delta_n = (n_e - n_o)/S0
    abt = (n_e + 2*n_o)/3.0
    n_e = abt + 2/3*s*delta_n
    n_o = abt - 1/3*s*delta_n
    return n_o, n_e
