import sys
import importlib
from os import path
import time
from typing import Self
import numpy as np
from scipy.integrate import quad,simpson

import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from PIL import Image,ImageOps
from matplotlib import cm

from ..utils.params import *
from ..utils.tools import *
from ..orderfield.lcsystems import *

class POMImage:

    def __init__(
            self,
            nlayers: int,
            Nx: int,
            Ny: int
    ):
        """
        Abstract class for the POM Intensity profiles with all common information
        :param nlayers: number of layers = number of wavelengths to compute
        """ 
        self.nlayers = nlayers      # Should be at least one wavelength to perform calculation
        self.Nx = Nx
        self.Ny = Ny
        self.Intensity =  np.zeros((Nx,Ny))

class SingleWave(POMImage):
    def __init__(self):
        """
        POMImage for single wavelength calculation
        """
        self.nlayers = 1
            
    
class MultiWave(POMImage):
    def __init__(self):
        """
        POMImage for multiple wavelength calculation 
        """
    
class LightSource:

    def __init__(self, spectrum: np.ndarray, reflect: str, source: str):
        """
        Characteristics of the incident light.
        Spectrum: an array with the wavelengths of the incident light
        Reflect: selects reflection attenuation mode: None, Fresnel, Empirical. Only valid for droplets
        Source:  specifies if the light is treated as uniform white light, LED lamp, or Halogen lamp
        """
        
        self.spectrum = spectrum
        self.reflect_mode = reflect
        self.source = source
        self.wavelength : float
    
class LED(LightSource):
        
        def __init__(self):

            """
            LED returns the intensity at wavelength x from the 
            spectrum of the microscope lamp fitted with several gaussian functions
            """
        
            self.source = "LED"

            self.y=0.15*gaussian (x, 0.45, 0.01)+0.41*gaussian (x, 0.525, 0.05)+0.37*gaussian (x, 0.625, 0.05) + 0.07*gaussian (x, 0.75, 0.05)
        

def light_xyz(self,wavelengths):
    """
    
    """
    lx = []; ly=[]; lz=[]
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
    """"
    Adjusting transmittance due to boundary of LC system and background
    """
    costheta_t = np.sqrt (1-n1/n2*np.sin(theta_i)**2)
    R_p = ((n1*costheta_t-n2*np.cos (theta_i))/(n1*costheta_t+n2*np.cos (theta_i)))**2
    R_s = ((n2*costheta_t-n1*np.cos (theta_i))/(n2*costheta_t+n1*np.cos (theta_i)))**2
    T_p = 1-R_p
    T_s = 1-R_s

    return T_p, T_s


#
#  ### Reference:
# Doane J. Appl. Phys. 69(9) 1991
# Microscope textures of nematic droplets in polymer dispersed liquid crystals
#
# Alberto J. Phys D: Appl. Phys.52 (2019) 213001
# Simulating optical polarizing microscopy textures using Jones calculus: a review exemplified with nematic liquid crystal tori
#

def rotation(alpha_p: float):
    """
    Jones matrix operator to rotate light around the optical axis.

    Args:
        alpha : angle of rotation around optical axis  [radians]
    Returns:
        3x3 matrix of the rotation operator           [-]
    """
    
    rot = np.asarray ([[np.cos (alpha_p), -np.sin(alpha_p), 0], 
                       [np.sin (alpha_p), np.cos(alpha_p), 0], 
                       [0,0,1]])

    return rot

def retardation(gamma: float, dphi: float):
    """
    Jones matrix operator to calculate the phase retardation.

    Args:
        gamma : angle of rotation around optical axis  [radians]
        dphi  : phase delay difference between ordinary and extraordinary axis
    Returns:
        2X2 matrix of the retardation operator           [-]
    """
                    
    Sr = np.eye(2, dtype = complex)

    CG = np.cos(gamma)
    SG = np.sin(gamma)

    ## I'm pretty sure my math is right on this one, but Elise please double check :)
    PP = np.exp(+dphi/2*1j)  # S_22
    QP = np.exp(-dphi/2*1j)  # S_11
    DP = np.sin(dphi/2) * 2j

    Sr[0][0] = CG*CG*PP + SG*SG*QP
    Sr[0][1] = -SG*CG*DP
    Sr[1][0] = -SG*CG*DP
    Sr[1][1] = SG*SG*PP + CG*CG*QP

    return Sr


def calculate_intensity (lc: LCGrid, inc_light: LightSource):
    """
    Calculates the resulting light intensity after it travels through an anisotropic media
    Information about the material is encoded in lc, and details about the inciden light
    are encoded in inc_light

    Args:
        lc        : Liquid crystal parameters and grid information
        inc_light : Incident light information, including the 'active' wavelength
    Returns:
        image     : SingleWave class that includes the Intensity map of the specified wavelength
    """
    #print ("Calculating light intensity")
    #print ("Refractive indices for wavelength: %d nm " %(wavelength*1000) )
    #print ("n_o = %.3f, n_e = %.3f, delta n = %.3f  "% (np.mean (n_o), np.mean(n_e), np.mean(n_e-n_o)))
    #print ("Applying Fresnel equation to calcuculate transmission?", toReflect)
    n2_avg = np.mean((2*lc.no+lc.ne)/3)
    n1 = 1.33

    # Identify nodes with non-zero director field.
    idx = np.linalg.norm(lc.director,axis=1) > 1.0E-3

    rot = rotation(inc_light.alpha)
    
    # Discretization parameters
    Nx = lc.grid.nl[0]
    Ny = lc.grid.nl[1]
    Nz = lc.grid.nl[2]

    image = SingleWave(Nx,Ny)

    for ix in range(Nx):

        if ((ix+1)%(int (Nx/10)) ==0):
                    print ("%d %%" %((ix+1) // (Nx/10)*10), end = '\t', flush = True)
        for iy in range(Ny):

            # Initialize
            Pold = np.eye(2,dtype=complex)
            #Sr = np.eye(2, dtype = complex)
            gamma0 = 0 # incident light 

            iiz2 = -1
            # Accumulates light retardation through LC media in the z direction
            for iz in range(Nz):
            #for iz in range(int (Nz*(0.5-0.05)),int (Nz*(0.5+0.05))):
                iiz = iz + iy*Nz + ix*Ny*Nz # the id of the cell

                if ( idx[iiz] ): # if the director is non-zero

                    director = lc.director[iiz]
                    director = np.matmul(rot, lc.director[iiz])
                    if ( director[2] < 0 ):
                        director[2] *= -1.0 # make it point in positive z

                    beta = np.arccos(director[2]) # angle between n_i and k_0
                    gamma1 = np.arctan2(director[1],director[0]) # angle between x and the projection of n_i
                    gamma = gamma1 - gamma0
                    #gamma = gamma0 - gamma1  # the gamma1 and the gamma0 are the alphas in the paper
                    #gamma0 = gamma1 # take this to be the next "incident light polarization"

                    if (hasS == False):
                        phio = 2*np.pi*lc.no*lc.grid.dx[2]/inc_light.wavelength # To calculate the rotation matrix
                        denom = [lc.no*np.sin(beta), lc.ne*np.cos(beta)]
                        nebeta = lc.no*lc.ne/np.linalg.norm(denom) #beta is gamma in the paper
                    else:
                        phio = 2*np.pi*lc.no[iiz]*lc.grid.dx[2]/inc_light.wavelength
                        denom = [lc.no[iiz]*np.sin(beta), lc.ne[iiz]*np.cos(beta)]
                        nebeta = lc.no[iiz]*lc.ne[iiz]/np.linalg.norm(denom) #beta is gamma in the paper

                    phie = 2*np.pi*nebeta*lc.grid.dx[2]/inc_light.waves

                    Sr = retardation(gamma, phio-phie )

                    Pnew = Sr @ Pold
                    Pold = Pnew
                    iiz2 = np.copy(iiz)

            ep = np.asarray([1,0], dtype = complex)
            ea = np.asarray([0,1], dtype = complex)
            res = np.matmul (ea, np.matmul (Pold, ep))
            image.Intensity[ix][iy] = np.real (np.conj(res)*res)

            if (iiz2>0 and inc_light.reflect_mode == "Fresnel"):
                xo, yo, zo = lc.grid.xyz[iiz2]
                theta_i = np.arcsin(np.sqrt ((xo**2 + yo**2)/(xo**2+yo**2+zo**2)))
                T1, T2 = Fresnel (theta_i, n1, n2_avg)
                trans = np.cos(theta_i)**2*T1**2 + np.sin(theta_i)**2*T2**2
                #print (theta_i*180/np.pi, trans)

                image.Intensity[ix][iy] = np.real (np.conj(res)*res)*trans*trans

    return image


def n_to_intensity(fname, wavelength, alpha_p, toReflect = True):

    wavelength = np.mean(wavelength)
    #Load data
    X = np.loadtxt(fname,dtype = np.float32);

    # If X has a 7 entries, then use the S parameters for calculating
    hasS = (X.shape[1] == 7)

    # Get refractive indices
    if (hasS):
        print ("Max and Mean of order parameter are: %.3f, %.3f" % (X[0,6],X[1,6]))
        ss = X[2:,6]
        n_o, n_e = calc_n_s (wavelength, ss)
    else:
        print ("No S data available, assume T= 25 Celsius")
        n_o, n_e = calc_n (wavelength)

    # Calculate image
    tmp =  calc_image (X, alpha_p = alpha_p, n_o = n_o, n_e = n_e, wavelength = wavelength, toReflect = toReflect)
    return tmp



def n_to_rgb_simp (fname, wavelengths = [0.65,0.55,0.45], alpha_p =0 , toReflect = True):

    #Load data
    X = np.loadtxt(fname,dtype = np.float32);

    # Test if file has S information
    hasS = (X.shape[1] == 7)
    if (hasS):
        print ("Max and Mean of order parameter are: %.3f, %.3f" % (X[0,6],X[1,6]))
        ss = X[2:,6]
    else:
        print ("No S data available, assume T= 25 Celsius")

    # Calculate image
    res =[]
    for wave in wavelengths:
        print ("%d" % (wave*1000), end = '\t')
        n_o, n_e = calc_n (wave)
        # Get refractive indices
        if (hasS):
            n_o, n_e = calc_n_s (wave, ss)
        else:
            n_o, n_e = calc_n (wave)
        res.append(  calc_image (X, alpha_p = alpha_p, n_o = n_o, n_e = n_e, wavelength = wave, reflect = toReflect))
    r = res[0]
    g = res[1]
    b = res[2]

    return np.asarray([r.T,g.T,b.T]).T


def calc_vmax(image1, image2):
    norms0= np.max (np.max(image1,axis =0),axis =0)
    norms45= np.max (np.max(image2,axis =0),axis =0)
    norms = np.max (np.asarray([norms0, norms45]),axis = 0)
    return norms


"""
input formats:
1. array
2. fname (ppng, .tiff, .jpeg)
3. PIL file

put the file destination for opening .png files
put array name for opening arrays

returns the array that has RGB values 0~255

Note: if it takes np array, it returns np array of the same size
"""

def RGB_to_BW (image,savename = None ):
    im1 = np.copy(image)
    # the function tends to mutate the original data
    #im1 = Image.open(r"C:\Users\System-Pc\Desktop\a.JPG")
    if (type (im1) == np.ndarray):
        if (np.median (im1)<1):
            im1 = np.asarray(im1, dtype = np.float32)
            im1 *= 255.0
        im1 = Image.fromarray(np.asarray(im1, dtype = np.uint8))

    elif (type(im1) == str):
        im1 = Image.open (im1)
    im2 = ImageOps.grayscale(im1)

    if (savename != None):
        plt.savefig(savename)
    im2 = np.asarray(im2)
    return np.asarray(im2/255.0, dtype = np.float32)


"""
Convert director field for multi wavelengths
"""
def n_to_color_manywaves(fname, wavelengths = np.arange(.400, .681, .02) , alpha_p =0, toReflect = True ):

    #Load data
    X = np.loadtxt(fname,dtype = np.float32);
    # If X has a 7 entries, then use the S parameters for calculating
    hasS = (X.shape[1] == 7)
    if (hasS):
        print ("Max and Mean of order parameter are: %.3f, %.3f" % (X[0,6],X[1,6]))
        ss = X[2:,6]
    else:
        print ("No S data available, assume T= 25 Celsius")
    # Calculate image
    res =[]
    for wave in wavelengths:
        print ("%d" % (wave*1000), end = '\t')
        n_o, n_e = calc_n (wave)
        # Get refractive indices
        if (hasS):
            n_o, n_e = calc_n_s (wave, ss)
        else:
            n_o, n_e = calc_n (wave)
        res.append(  calc_image (X, alpha_p = alpha_p, n_o = n_o, n_e = n_e, wavelength = wave, toReflect = toReflect))
    res = np.asarray(res)
    res = np.transpose (res, [1,2,0])
    res2 = np.copy (res)

    #print ("Angle %d" % int(180*alpha_p/np.pi))
    return res2 # pixels_y *pixels_x * N_waves


def n_to_rgb_full(fname, wavelengths = np.arange(.400, .681, .02), angle = 0, exposureFactor =1.0, toReflect = True):
    print ("fname", fname, "angle:", angle)
    print("Number of wavelengths", len(wavelengths)-1)
    midwaves, ws = light_xyz(wavelengths)
    #wavelenths = np.copy(midwaves)
    images_cont = n_to_color_manywaves (fname, wavelengths = midwaves, alpha_p = angle, toReflect = toReflect)
    #ws = cie_xyz(wavelengths) # Calculates the relative weights that integrate into the 3 color channels

    # White-balance
    #ws2 = ws/np.sum(ws)*3 # Normalize (divide by 3 since each sums to 1)
    s1, s2, s3 = white_balance (ws, exposureFactor = exposureFactor)
    ws2 = np.copy(ws)
    ws2[:,0]*= s1; ws2[:,1]*= s2; ws2[:,2]*= s3

    image_xyz = np.matmul(images_cont, ws2) # Cast from "continuous" spectrum to XYZ channels

    tmp = xyz2rgb(image_xyz)# Convert XYZ to RGB image See Wikipedia

    #plot_hist_rgb(image_xyz)
    # Although it's mostly normalized,xyz_to_rgb causes image to slightly go out of [0,1]
    # Normalize to avoid saturation

    #imax = np.max(np.max(tmp, axis =0),axis=0)
    #print ("Max of three color chanels", imax)
    #imax [np.where (imax<1)] =1
    #imin = np.min(np.min(tmp, axis =0),axis=0)
    #res = (1/ (imax) )*(tmp) # Normalize
    idx = np.where(tmp>1)
    res = np.copy(tmp)
    res[idx] = 1.0
    res[np.where(res<0)] = 0

    del tmp
    del images_cont
    del image_xyz
    del angle

    return res


# # Let's start to make POM images!


"""
frame input types:
1. int
    that corresponds to a frame in an animation
2. string
    that corresponds to the file name of the interpolated director field
"""
def POMFrame (frame, mode, angle, wl = None, exposureFactor = 1.0,toReflect1 = True):

    print ("="*100)
    print ("Calculation started")
    print ("="*100)
    time1 = time.time()
    directory1 = "./Interpolated_Director_Field/"
    directory2 = "./Images/"
    angle = angle*np.pi/180.0

    angle1 = np.copy(angle)
    exposureFactor1 = np.copy(exposureFactor)
    # Load
    if (type(frame) == int):
        fname = directory1+ "Frame-"+str(frame)+"-interpolated-directors.txt"
        info = directory2+"Frame-"+str(frame)
    elif (type(frame) == str):
        fname = directory1+frame
        n2 = path.splitext(frame)[0]
        info = directory2+n2
    else:
        print ("Error: wrong filename")
        return

    # Calculate images according to mode
    if (mode == "Single-wavelength"):

        #wave = wl
        image = n_to_intensity(fname, wavelength = wl, alpha_p = angle1, toReflect = toReflect1)
        #image = n_to_rgb_full (fname,wavelengths = wl, angle= angle1, exposureFactor = exposureFactor1, toReflect = toReflect1)

        ## Plot it
        picname = info+"-angle-"+str(int(180*angle1/np.pi)) +"-lambda-"+str(int(np.mean(wl)*1000))+".png"
        plot_image(image,vmax = 1.0, savename = picname)
        picname = info+"-angle-"+str(int(180*angle1/np.pi)) +"-lambda-"+str(int(np.mean(wl)*1000))+"Hist.png"
        plot_hist (image,savename = picname)

    if (mode == "Simp-color"):
        print ("Naive RGB image calculations")
        # Calculate RGB images
        image_rgb = n_to_rgb_full (fname,wavelengths = wl, angle= angle1, exposureFactor = exposureFactor1, toReflect = toReflect1)
        # RGB channel plots
        picname = info+"-angle-"+str(int(180*angle1/np.pi)) +"-SimpRGB-channels.png"
        plot_image_rgb(image_rgb,vmax = 1.0,savename = picname)
        # RGB histograms
        picname = info+"-angle-"+str(int(180*angle1/np.pi)) +"-SimpRGB-hist.png"
        plot_hist_rgb (image_rgb, savename=picname)
        # RGB images
        picname = info+"-angle-"+str(int(180*angle1/np.pi)) +"-SimpRGB.png"
        plot_image(image_rgb,vmax = 1.0,savename = picname)

    if (mode == "Full-color"):

        print ("RGB image from multiple wavelengths")

        # Initialize continuous wavelengths
        if wl is None:
            print ("Default wavelengths")
            wl = np.arange(.400, .681, .014)


         # Calculate images
        image_rgbf0 = n_to_rgb_full (fname,wavelengths = wl, angle= angle1, exposureFactor = exposureFactor1, toReflect = toReflect1)

        # Plot RGB images
        picname = info+"-angle-"+str(int(180*angle1/np.pi)) +"-FullRGB.png"
        plot_image(image_rgbf0 ,vmax = 1.0, savename = picname)

        # RGB channel plots
        picname = info+"-angle-"+str(int(180*angle1/np.pi)) +"-FullRGB-channels.png"
        plot_image_rgb(image_rgbf0,vmax = 1.0,savename = picname)

        # Plot histograms
        picname = info+"-angle-"+str(int(180*angle1/np.pi)) +"-FullRGB-Hist.png"
        plot_hist_rgb(image_rgbf0, savename = picname)

        # Plot BW
        picname = info+"-angle-"+str(int(180*angle1/np.pi)) +"-FullRGB"+"-BW.png"
        plot_image(RGB_to_BW(image_rgbf0),vmax= 1.0, savename = picname)

        # Save .npy files
        npyname = info+"-angle-"+str(int(180*angle1/np.pi)) +"-FullRGB"+".npy"
        np.save (npyname, image_rgbf0)

        time2 = time.time()
        t = time2-time1
        print ("Elapsed time: %.1f s \n" % t)
    plt.close ("all")
    return
