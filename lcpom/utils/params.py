import numpy as np
from scipy.integrate import simpson
from ..utils.tools import *

#
#  Refractive indices
#  The refractive indices depend on wavelengths (and temperature).
#
#  Reference:
#
#  WU et al. Optical Engineering 1993 32(8) 1775
#  Li et al. Journal of Applied Physics 96, 19 (2004)


def calc_n(lamb):
    """
    calc_n calculates the refractive indices (n_o, n_e) of 5CB for the wavelength lamb
    """
    l1 = 0.210; l2 = 0.282;
    n0e = 0.455; g1e = 2.325; g2e = 1.397
    n0o = 0.414; g1o = 1.352; g2o = 0.470

    n_e = 1 + n0e + g1e*(lamb**2 * l1**2)/(lamb**2-l1**2) + g2e*(lamb**2 * l2**2)/(lamb**2-l2**2)
    n_o = 1 + n0o + g1o*(lamb**2 * l1**2)/(lamb**2-l1**2) + g2o*(lamb**2 * l2**2)/(lamb**2-l2**2)

    return n_o, n_e

def calc_n_s(lamb,s):
    """
    calc_n_s calculates the refractive indices (n_o,n_e) of 5CB for the wavelength lamb
    and order parameter s
    """
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

def LED(x):
    """
    LED returns the intensity at wavelength x from the 
    spectrum of the microscope lamp fitted with several gaussian functions
    """
    y=0.15*gaussian (x, 0.45, 0.01)+0.41*gaussian (x, 0.525, 0.05)+0.37*gaussian (x, 0.625, 0.05) + 0.07*gaussian (x, 0.75, 0.05)
    return y

def light_xyz(wavelengths):
    """
    
    """
    lx = []; ly=[];lz=[]
    wv = []
    dl = wavelengths[1]-wavelengths[0]
    
    for i in range (0, len (wavelengths)-1):
        start = wavelengths[i]
        end = wavelengths[i+1]
        wv.append (start + 0.5*dl)
        x = np.linspace (start, end, 20)
        light = LED(x)
        res = cie_xyz(x)
        res[:,0]*= light;res[:,1]*= light;res[:,2]*= light
        lx.append(simpson (res[:,0],x))
        ly.append(simpson (res[:,1],x))
        lz.append(simpson (res[:,2],x))
    
    lx = np.asarray(lx)
    ly = np.asarray(ly)
    lz = np.asarray(lz)
    wv = np.asarray(wv)
    res = np.vstack([lx,ly,lz]).T
    
    return wv, res

def cie_xyz(wv):
    """
    
    """
    waves = np.copy(wv)
    if (np.mean(wv))<10:
        #print("rescale units um to nm")
        waves*=1000
    wx = 1.056*g_p(waves, 599.8, 37.9, 31.0)+0.362*g_p(waves, 442.0, 16.0, 26.7)-0.065*g_p(waves, 501.1, 20.4, 26.2)
    wy = 0.821*g_p(waves, 568.8, 46.9, 40.5)+0.286*g_p(waves, 530.9, 16.3, 31.1)
    wz = 1.217*g_p(waves, 437.0, 11.8, 36.0)+0.681*g_p(waves, 459.0, 26.0, 13.8)
    res = np.asarray([wx, wy, wz]).T
    return res

def Fresnel(theta_i, n1, n2 ):
    costheta_t = np.sqrt (1-n1/n2*np.sin(theta_i)**2)
    R_p = ((n1*costheta_t-n2*np.cos (theta_i))/(n1*costheta_t+n2*np.cos (theta_i)))**2
    R_s = ((n2*costheta_t-n1*np.cos (theta_i))/(n2*costheta_t+n1*np.cos (theta_i)))**2
    T_p = 1-R_p
    T_s = 1-R_s
    #T_s = T_s*(T_s>0)
    return T_p, T_s

def white_balance(ws, whiteRGB = np.asarray([1.0, 1.0, 1.0]), exposureFactor = 1.0):
    print ("Exposure factor is:", exposureFactor)
    #x0 = 0.964; y0 = 1.000; z0 = 0.825
    x0, y0, z0 = rgb2xyz(np.asarray (whiteRGB).reshape(1,1,3)).reshape(3)
    x0, y0, z0 = np.asarray([0.95046, 1.     , 1.08875])
    s1 = x0/sum(ws[:,0])*exposureFactor; s2 = y0/sum(ws[:,1])*exposureFactor; s3 = y0/sum(ws[:,2])*exposureFactor
    print ("White balance scaling factor: %.2f, %.2f, %.2f" % (s1, s2, s3))

    return s1, s2, s3
