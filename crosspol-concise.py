#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import path
from matplotlib import cm
import matplotlib

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

cmap = plt.get_cmap('jet')
plt.style.use('./large_plot.mplstyle')

from PIL import Image,ImageOps
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


#
#  ### Reference:
# Doane J. Appl. Phys. 69(9) 1991
# Microscope textures of nematic droplets in polymer dispersed liquid crystals
#
# Alberto J. Phys D: Appl. Phys.52 (2019) 213001
# Simulating optical polarizing microscopy textures using Jones calculus: a review exemplified with nematic liquid crystal tori
#

# In[ ]:



def calc_image (X, alpha_p , n_o , n_e , wavelength):
    print ("Calculating light intensity")
    print ("Refractive indices for wavelength: %d nm " %(wavelength*1000) )
    print ("n_o = %.3f, n_e = %.3f, delta n = %.3f  "% (np.mean (n_o), np.mean(n_e), np.mean(n_e-n_o)))
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
                    print ("%d %%" %((ix+1) // (Nx/10)*10), end = '\t')
        for iy in range(Ny):

            # Initialize

            Pold = np.eye(2,dtype=complex)
            Sr = np.eye(2, dtype = complex)
            gamma0 = 0 # incident light direction


            for iz in range(Nz):
            #for iz in range(int (Nz*(0.5-0.05)),int (Nz*(0.5+0.05))):
                iiz = iz + iy*Nx + ix*Ny*Nz # the id of the cell
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

            ep = np.asarray([1,0], dtype = complex)
            ea = np.asarray([0,1], dtype = complex)
            #ep = np.matmul (rot2, ep)
            #ea = np.matmul(rot2, ea)
            res = np.matmul (ea, np.matmul (Pold, ep))

            Intensity[ix][iy] = np.real (np.conj(res)*res)

    print("\n")

    return Intensity


# In[ ]:



# # Refractive index
# The refractive indices depend on wavelengths (and temperature).
#
# Reference:
#
# WU et al. Optical Engineering 1993 32(8) 1775
# Li et al. Journal of Applied Physics 96, 19 (2004)
# In[4]:


# In[ ]:


def calc_n(lamb):

    l1 = 0.210; l2 = 0.282;
    n0e = 0.455; g1e = 2.325; g2e = 1.397
    n0o = 0.414; g1o = 1.352; g2o = 0.470

    n_e = 1 + n0e + g1e*(lamb**2 * l1**2)/(lamb**2-l1**2) + g2e*(lamb**2 * l2**2)/(lamb**2-l2**2)
    n_o = 1 + n0o + g1o*(lamb**2 * l1**2)/(lamb**2-l1**2) + g2o*(lamb**2 * l2**2)/(lamb**2-l2**2)

    return n_o, n_e


# In[ ]:


def calc_n_s(lamb,s):

    l1 = 0.210; l2 = 0.282;
    n0e = 0.455; g1e = 2.325; g2e = 1.397
    n0o = 0.414; g1o = 1.352; g2o = 0.470

    n_e = 1 + n0e + g1e*(lamb**2 * l1**2)/(lamb**2-l1**2) + g2e*(lamb**2 * l2**2)/(lamb**2-l2**2)
    n_o = 1 + n0o + g1o*(lamb**2 * l1**2)/(lamb**2-l1**2) + g2o*(lamb**2 * l2**2)/(lamb**2-l2**2)

    S0 = 0.76
    delta_n = (n_e - n_o)/S0
    abt = (n_e + 2*n_o)/3.0
    n_e = abt + 2/3*s*delta_n
    n_o = abt - 1/3*s*delta_n
    return n_o, n_e


# In[ ]:



def n_to_intensity(fname, wavelength, alpha_p):

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
    tmp =  calc_image (X, alpha_p = alpha_p, n_o = n_o, n_e = n_e, wavelength = wavelength)
    return tmp


# In[ ]:


def plot_image (intensity, vmax = None,savename = None):
    fig, ax = plt.subplots()
    image = intensity
    if (vmax == None):
        vmax = np.max(image)
    im = ax.imshow(image, cmap=plt.get_cmap('bone'),interpolation='bicubic',vmax = vmax)
    #ax.set_title ("0$^o$")
    ax.set_ylim(0,image.shape[0]-1)
    ax.set_xlim(0,image.shape[1]-1)
    im.axes.get_xaxis().set_visible(False);
    im.axes.get_yaxis().set_visible(False);
    ax.axis("off")
    plt.tight_layout(pad = 0)
    dpi = matplotlib.rcParams['savefig.dpi']
    fig.set_size_inches(5*image.shape[1]/dpi,5*image.shape[0]/dpi)
    if (savename != None):
        plt.savefig(savename,pad_inches=0)
    return


# In[ ]:



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
        im = ax.imshow(image, cmap=plt.get_cmap(color_maps[i]),interpolation='bicubic',vmin = 0,vmax = vmax)
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


# In[ ]:


def plot_hist (ys, savename=None):
    fig, ax = plt.subplots()
    ys = np.asarray(ys)

    upper = np.max(ys)
    if (upper<1.0E-2):
        upper = 1.0
    image = ys
    #for image in ys:
    ax.hist(image.flatten(), bins = np.linspace (0,upper,51), density = True);
    ax.set_yscale ("log")
    ax.set_xlabel("Intensity")
    plt.tight_layout()
    if (savename != None):
        plt.savefig(savename)
    return


# In[ ]:


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


# In[ ]:


def n_to_rgb_simp (fname, wavelengths = [0.65,0.55,0.45], alpha_p =0 ):

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
        res.append(  calc_image (X, alpha_p = alpha_p, n_o = n_o, n_e = n_e, wavelength = wave))
    r = res[0]
    g = res[1]
    b = res[2]

    return np.asarray([r.T,g.T,b.T]).T


# In[11]:


def calc_vmax(image1, image2):
    norms0= np.max (np.max(image1,axis =0),axis =0)
    norms45= np.max (np.max(image2,axis =0),axis =0)
    norms = np.max (np.asarray([norms0, norms45]),axis = 0)
    return norms


# In[12]:


# In[ ]:



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
    im1 = np.copy(image) # the function tends to mutate the original data
    #im1 = Image.open(r"C:\Users\System-Pc\Desktop\a.JPG")
    if (type (im1) == np.ndarray):
        if (np.median (im1)<1):
            im1 *= 255.0
        im1 = Image.fromarray(np.asarray(im1, dtype = np.uint8))

    elif (type(im1) == str):
        im1 = Image.open (im1)
    im2 = ImageOps.grayscale(im1)

    if (savename != None):
        plt.savefig(savename)
    im2 = np.asarray(im2)
    return np.asarray(im2/255.0, dtype = np.float32)


# In[ ]:


def g_p(x, mu, sig1, sig2):
    y = (x<mu)*np.exp(-(x-mu)**2/ (2*sig1**2)) + (x>=mu)*np.exp(-(x-mu)**2/ (2*sig2**2))
    return y


# In[ ]:





# In[14]:


def cie_xyz(wv):
    waves = np.copy(wv)
    if (wv<1).any() :
        #print("rescale units um to nm")
        waves*=1000
    wx = 1.056*g_p(waves, 599.8, 37.9, 31.0)+0.362*g_p(waves, 442.0, 16.0, 26.7)-0.065*g_p(waves, 501.1, 20.4, 26.2)
    wy = 0.821*g_p(waves, 568.8, 46.9, 40.5)+0.286*g_p(waves, 530.9, 16.3, 31.1)
    wz = 1.217*g_p(waves, 437.0, 11.8, 36.0)+0.681*g_p(waves, 459.0, 26.0, 13.8)
    res = np.asarray([wx, wy, wz]).T
    return res


# In[15]:


"""
These functions are copied from the mahotas package
"""
def _convert(array, matrix, dtype, funcname):
    h,w,d = array.shape
    array = array.transpose((2,0,1))
    array = array.reshape((3,h*w))
    array = np.dot(matrix, array)
    array = array.reshape((3,h,w))
    array = array.transpose((1,2,0))
    if dtype is not None:
        array = array.astype(dtype, copy=True)
    return array


# In[ ]:


def xyz2rgb(xyz, dtype=None):
    '''
    rgb = xyz2rgb(xyz, dtype={float})
    Convert XYZ to sRGB coordinates
    The output should be interpreted as sRGB. See Wikipedia for more details:
    http://en.wikipedia.org/wiki/SRGB
    Parameters
    ----------
    xyz : ndarray
    dtype : dtype, optional
        What dtype to return. Default will be floats
    Returns
    -------
    rgb : ndarray
    See Also
    --------
    rgb2xyz : function
        The reverse function
    '''
    transformation = np.array([
                [ 3.2406, -1.5372, -0.4986],
                [-0.9689,  1.8758,  0.0415],
                [ 0.0557, -0.2040,  1.0570],
                ])

    rgb_linear = _convert(xyz, transformation, dtype, 'xyz2rgb')


    """
    a = 0.055
    srgb_high = (1 + a)*np.power(rgb_linear, 1./2.4)
    srgb_high -= a
    srgb_low = 12.92 * rgb_linear
    srgb = np.choose(rgb_linear <= 0.0031308, [srgb_low, srgb_high])
    srgb *= 255.
    srgb = np.asarray(srgb,dtype = np.int32)
    """
    return rgb_linear


# In[ ]:


"""
Convert director field for multi wavelengths
"""
def n_to_color_manywaves(fname, wavelengths = np.arange(.400, .681, .02) , alpha_p =0 ):

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
        res.append(  calc_image (X, alpha_p = alpha_p, n_o = n_o, n_e = n_e, wavelength = wave))
    res = np.asarray(res)
    res = np.transpose (res, [1,2,0])
    res2 = np.copy (res)


    #print ("Angle %d" % int(180*alpha_p/np.pi))
    return res2 # pixels_y *pixels_x * N_waves


# In[ ]:


def n_to_rgb_full(fname, wavelengths = np.arange(.400, .681, .02), angle = 0):
    print ("fname", fname, "angle:", angle)
    print("Number of wavelengths", len(wavelengths))
    images_cont = n_to_color_manywaves (fname, wavelengths, alpha_p = angle)
    ws = cie_xyz(wavelengths) # Calculates the relative weights that integrate into the 3 color channels
    ws2 = ws/np.sum(ws)*3 # Normalize (divide by 3 since each sums to 1)
    image_xyz = np.matmul(images_cont, ws2) # Cast from "continuous" spectrum to XYZ channels

    tmp = xyz2rgb(image_xyz)# Convert XYZ to RGB image See Wikipedia

    #plot_hist_rgb(image_xyz)
    # Although it's mostly normalized,xyz_to_rgb causes image to slightly go out of [0,1]
    # Normalize to avoid saturation

    imax = np.max(np.max(tmp, axis =0),axis=0)
    print ("Max of three color chanels", imax)
    imax [np.where (imax<1)] =1
    #imin = np.min(np.min(tmp, axis =0),axis=0)
    res = (1/ (imax) )*(tmp) # Normalize
    res[np.where(res<0)] = 0

    del tmp
    del images_cont
    del image_xyz
    del angle

    return res



# # Let's start to make POM images!

# In[ ]:


mode = num_to_mode(3)
angle = 110
#POM_of_Frame("test.txt", mode,angle)
POM_of_Frame("testS.txt", mode,angle)


# In[ ]:




"""
frame input types:
1. int
    that corresponds to a frame in an animation
2. string
    that corresponds to the file name of the interpolated director field
"""
def POM_of_Frame (frame, mode ,angle , wl = None):
    directory1 = "./Interpolated_Director_Field/"
    directory2 = "./Images/"
    angle = angle*np.pi/180.0

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
        print ("Single wavelength calculations")
        if wl is None:
            wl = input ("Please input wavelengths in microns: (0.4~0.68 for visible light) \t")
            wl = float(wl)
        if ((wl > 0.35 and wl < 0.7) == False):
            print ("Invalid wavelength")
        else:
            print (wl)
        wave = wl

        image = n_to_intensity(fname, wave, angle)

        ## Plot it
        picname = info+"-angle-"+str(int(180*angle/np.pi)) +"-lambda-"+str(int(wave*1000))+".png"
        plot_image(image,vmax = 1.0, savename = picname)
        picname = info+"-angle-"+str(int(180*angle/np.pi)) +"-lambda-"+str(int(wave*1000))+"Hist.png"
        plot_hist (image,savename = picname)

    if (mode == "Simp-color"):
        print ("Naive RGB image calculations")
        # Calculate RGB images
        image_rgb = n_to_rgb_simp (fname, alpha_p = angle)
        # RGB channel plots
        picname = info+"-angle-"+str(int(180*angle/np.pi)) +"-SimpRGB-channels.png"
        plot_image_rgb(image_rgb,vmax = 1.0,savename = picname)
        # RGB histograms
        picname = info+"-angle-"+str(int(180*angle/np.pi)) +"-SimpRGB-hist.png"
        plot_hist_rgb (image_rgb, savename=picname)
        # RGB images
        picname = info+"-angle-"+str(int(180*angle/np.pi)) +"-SimpRGB.png"
        plot_image(image_rgb,vmax = 1.0,savename = picname)

    if (mode == "Full-color"):

        print ("RGB image from multiple wavelengths")

        # Initialize continuous wavelengths
        if wl is None:
            print ("Default wavelengths")
            wl = np.arange(.400, .681, .02)

        angle1 = angle

         # Calculate images
        image_rgbf0 = n_to_rgb_full (fname,wavelengths = wl, angle= angle1)

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

    return


def num_to_mode (num):
    if (num == 1):
        return "Single-wavelength"
    if (num ==2):
        return "Simp-color"
    if (num == 3):
        return "Full-color"

    else: return 0
# In[72]:
if __name__ == "__main__":

    case = input ("Please select input mode. [1-3]\n 1. Single image.\n 2.Batch processing.\n\t Names shall be specified in ./tmp-filenames.txt. \n\t The exact director files need to to stored in 'Interpolated_Director_Fields' folder.\n 3. Batch processing specified by frames. The frames are listed in 'tmp-frames.txt'. \n ")
    case = int (case)
    if (isinstance (case, int) == False):
        sys.exit("Case is not integer")

    angle = input ("Angle of polarizer in degrees [0 - 180]\n")
    angle = float(angle)
    if ( (angle <0) or (angle>180) ):
        sys.exit("Angle out of range is not integer")
    num = input ("Select color mode [1-3]: 1. Single wavelength 2. Simplified color 3. Full color\n")
    num = int(num)
    if ( isinstance (num, int) == False):
        sys.exit("num is not integer")
    mode = num_to_mode(num)


    # Generate images according to case
    if (case == 1):
        # # Single image

        name = input ("File location for the director field. *.txt \n")
        POM_of_Frame(name, mode,angle)

    elif (case ==2):
        # # Batch by filenames
        # Compute for all files listed in "./tmp-filenames.txt" (the exact director files need to to stored in "Interpolated_Director_Fields" folder)



        with open("./tmp-filenames.txt") as fp:
            for name in fp:
                if (num == 1):
                    POM_of_Frame(name.strip('\n'), mode,angle)
                else:
                    POM_of_Frame(name.strip('\n'), mode, angle,wl = np.arange(.400, .681, .014))


    elif (case ==3):
        # # Batch by frames
        # The frames are listed in "tmp-frames.txt"

        frames= np.loadtxt("tmp-frames.txt", dtype = np.int32)

        for frame in frames:
            #POM_of_Frame(frame, mode, angle)
            POM_of_Frame(frame, mode, angle,wl = np.arange(.400, .681, .014))
    else:
        print ("Error, wrong case.")
