import sys
import importlib
from os import path
import time
import numpy as np
from scipy.integrate import quad,simpson

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from PIL import Image,ImageOps
from matplotlib import cm

from ..utils.params import *
from ..utils.tools import *

class POMImage:

    def __init__(self,nlayers=None):
        """
        Abstract class for the POM Intensity profiles with all common information
        :param nlayers: number of layers = number of wavelengths to compute
        """
        self.nlayers = nlayers

class SingleWave(POMImage):
    def __init__(self):
        """
        POMImage for single wavelength calculation
        """
    
class MultWave(POMImage):
    def __init__(self):
        """
        POMImage for multiple wavelength calculation 
        """

    



#
#  ### Reference:
# Doane J. Appl. Phys. 69(9) 1991
# Microscope textures of nematic droplets in polymer dispersed liquid crystals
#
# Alberto J. Phys D: Appl. Phys.52 (2019) 213001
# Simulating optical polarizing microscopy textures using Jones calculus: a review exemplified with nematic liquid crystal tori
#


def calc_image (X, alpha_p , n_o , n_e , wavelength, toReflect):
    print ("Calculating light intensity")
    print ("Refractive indices for wavelength: %d nm " %(wavelength*1000) )
    print ("n_o = %.3f, n_e = %.3f, delta n = %.3f  "% (np.mean (n_o), np.mean(n_e), np.mean(n_e-n_o)))
    print ("Applying Fresnel equation to calcuculate transmission?", toReflect)
    n2_ave = np.mean((2*n_o+n_e)/3)
    n1 = 1.33

    # the header two lines
    [Nx, Ny, Nz] = np.asarray (X[0, :3], dtype = np.int32)
    [dx, dy, dz] = X[0, 3:6]
    [x_min, x_max, y_min, y_max, z_min, z_max] = X[1,0:6]
    print ("Data shape: ", X.shape)
    print ("Number of data points:", Nx*Ny*Nz)
    print ("dx = %.2f" %(dx))

    # the actual data
    rr = X[2:,:3]; nn = X[2:,3:6]

    # Determine if the input file has S information
    hasS = False
    if (X.shape[1] == 7):
        hasS = True
        print ("Data S information, n_e and n_o has spatial variation.")


    # The non-zero n
    idx = np.linalg.norm(nn,axis=1) > 1.0E-3

    rot = np.asarray ([[np.cos (alpha_p), -np.sin(alpha_p),0],[np.sin (alpha_p), np.cos(alpha_p),0],[0,0,1]])
    #rot2 =np.asarray ([[np.cos (alpha_p), -np.sin(alpha_p)],[np.sin (alpha_p), np.cos(alpha_p)]])
    Intensity = np.zeros((Nx,Ny))
    for ix in range(Nx):

        if ((ix+1)%(int (Nx/10)) ==0):
                    print ("%d %%" %((ix+1) // (Nx/10)*10), end = '\t', flush = True)
        for iy in range(Ny):

            # Initialize
            Pold = np.eye(2,dtype=complex)
            Sr = np.eye(2, dtype = complex)
            gamma0 = 0 # incident light direction

            iiz2 = -1
            for iz in range(Nz):
            #for iz in range(int (Nz*(0.5-0.05)),int (Nz*(0.5+0.05))):
                iiz = iz + iy*Nz + ix*Ny*Nz # the id of the cell

                if ( idx[iiz] ): # if the director is non-zero

                    director = nn[iiz]
                    director = np.matmul(rot, nn[iiz])
                    if ( director[2] < 0 ):
                        director[2] *= -1.0 # make it point in positive z

                    beta = np.arccos(director[2]) # angle between n_i and k_0
                    gamma1 = np.arctan2(director[1],director[0]) # angle between x and the projection of n_i
                    gamma = gamma1 - gamma0
                    #gamma = gamma0 - gamma1  # the gamma1 and the gamma0 are the alphas in the paper
                    #gamma0 = gamma1 # take this to be the next "incident light polarization"

                    if (hasS == False):
                        phio = 2*np.pi*n_o*dz/wavelength # To calculate the rotation matrix
                        denom = [n_o*np.sin(beta), n_e*np.cos(beta)]
                        nebeta = n_o*n_e/np.linalg.norm(denom) #beta is gamma in the paper
                    else:
                        phio = 2*np.pi*n_o[iiz]*dz/wavelength
                        denom = [n_o[iiz]*np.sin(beta), n_e[iiz]*np.cos(beta)]
                        nebeta = n_o[iiz]*n_e[iiz]/np.linalg.norm(denom) #beta is gamma in the paper


                    phie = 2*np.pi*nebeta*dz/wavelength
                    cs_phie = np.cos(phie) + 1j*np.sin(phie) # S_22
                    cs_phio = np.cos(phio) + 1j*np.sin(phio) # S_11

                    Sr[0][0] = np.power(np.cos(gamma),2)*cs_phie  + np.power(np.sin(gamma),2)*cs_phio
                    Sr[0][1] = -np.sin(gamma)*np.cos(gamma)*( cs_phie - cs_phio )
                    Sr[1][0] = -np.sin(gamma)*np.cos(gamma)*( cs_phie - cs_phio )
                    Sr[1][1] = np.power(np.sin(gamma),2)*cs_phie  + np.power(np.cos(gamma),2)*cs_phio

                    Pnew = np.matmul(Sr,Pold)
                    Pold = Pnew
                    iiz2 = np.copy(iiz)

            ep = np.asarray([1,0], dtype = complex)
            ea = np.asarray([0,1], dtype = complex)
            #ep = np.matmul (rot2, ep)
            #ea = np.matmul(rot2, ea)
            res = np.matmul (ea, np.matmul (Pold, ep))
            Intensity[ix][iy] = np.real (np.conj(res)*res)
            if (iiz2>0 and toReflect == True):
                xo, yo, zo = rr[iiz2]
                theta_i = np.arcsin(np.sqrt ((xo**2 + yo**2)/(xo**2+yo**2+zo**2)))
                T1, T2 = Fresnel (theta_i, n1, n2_ave)
                trans = np.cos(theta_i)**2*T1**2 + np.sin(theta_i)**2*T2**2
                #print (theta_i*180/np.pi, trans)
                Intensity[ix][iy] = np.real (np.conj(res)*res)*trans*trans

    print("\n")

    return Intensity


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


def plot_image_rgb (image_rgb, vmax = None,savename = None):
    fig, axes = plt.subplots(1,3, sharey = True)
    color_maps = ["Reds", "Greens", "Blues"] # These are the three color map keys
    if (vmax == None):
            print ("vmax not specified. set auto vmax")
            vmax = np.max(image_rgb)
            if (vmax>0.8 and vmax < 1.05):
                vmax = 1.0
            print ("vmax =", vmax)
    for i in range (3):
        ax = axes[i]
        image = image_rgb[:,:,i]
        image = np.transpose(image)
        im = ax.imshow(image, cmap=plt.get_cmap(color_maps[i]),interpolation='bicubic',origin = 'lower',vmin = 0,vmax = vmax)
        #ax.set_title ("0$^o$")
        ax.set_ylim(0,image.shape[0]-1) # Seems that -1 is necessary?
        ax.set_xlim(0,image.shape[1]-1)
        #ax.axis("off")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    dpi = matplotlib.rcParams['savefig.dpi']
    fig.set_size_inches(3*5*image.shape[1]/dpi,5*image.shape[0]/dpi)

    plt.tight_layout(pad=0)
    if (savename != None):
        plt.savefig(savename)

    return



def plot_hist_rgb (ys, savename=None):
    fig, axes = plt.subplots(3,1,figsize = (5,5),sharex = True)
    colors = [[1,0,0],[0,1,0],[0,0,1]]
    #m = np.log10(np.max(ys.flatten()))
    ys = np.asarray(ys)
    upper = np.max(ys)
    if (upper<1.0E-2):
        upper = 1.0
    for i in np.arange (2,-1,-1):
        ax = axes[i]
        image = ys[:,:,i]
        ax.hist(image.flatten(), color= colors[i],bins = np.linspace (0,upper,51), density = True);
        ax.set_yscale ("log")
    #ax.set_xscale ("log")
    axes[2].set_xlabel("Intensity")
    plt.tight_layout()
    if (savename != None):
        plt.savefig(savename)
    return


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
        res.append(  calc_image (X, alpha_p = alpha_p, n_o = n_o, n_e = n_e, wavelength = wave, toReflect = toReflect))
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
