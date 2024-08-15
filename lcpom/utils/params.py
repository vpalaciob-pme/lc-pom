import numpy as np
from scipy.integrate import simpson
from ..utils.tools import *

from plum import dispatch

#
#  Refractive indices
#  The refractive indices depend on wavelengths (and temperature).
#
#  Reference:
#
#  WU et al. Optical Engineering 1993 32(8) 1775
#  Li et al. Journal of Applied Physics 96, 19 (2004)

def white_balance(ws, whiteRGB = np.asarray([1.0, 1.0, 1.0]), exposureFactor = 1.0):
    """
    
    """
    #print ("Exposure factor is:", exposureFactor)
    #x0 = 0.964; y0 = 1.000; z0 = 0.825
    x0, y0, z0 = rgb2xyz(np.asarray (whiteRGB).reshape(1,1,3)).reshape(3)
    x0, y0, z0 = np.asarray([0.95046, 1.     , 1.08875])
    s1 = x0/sum(ws[:,0])*exposureFactor; s2 = y0/sum(ws[:,1])*exposureFactor; s3 = y0/sum(ws[:,2])*exposureFactor
    #### QUESTION FOR ELISE. error in s3? should it be z0?
    #print ("White balance scaling factor: %.2f, %.2f, %.2f" % (s1, s2, s3))

    return s1, s2, s3
