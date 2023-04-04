#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import numpy.linalg as la
import re

cmap = plt.get_cmap('jet')
plt.style.use('./large_plot.mplstyle')
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


# # Generate a vector field

# In[4]:


def ellip1 (r, L,center):
    x = (r[0]-center[0])*2/L[0]
    y = (r[1]-center[1])*2/L[1]
    z = (r[2]-center[2])*2/L[2]
    return x**2+y**2 +z**2-1


# In[5]:


"""
Generates radial configuration
Inputs
r: radius of the sphere
dx: interval of grid

"""
def radial (r,  dx = 0.1):
    # Make sphere centered at origin
    L = 2*r*np.ones(3);
    centroid = np.asarray([0,0,0])
    # Pad by 10%, round nx, ny, nz to be multiples of 10
    l_box = L *1.1
    l_box[2] = L[2]*1.01
    nl = np.ceil(l_box/dx/10)*10
    nl = np.asarray (nl, dtype= np.int32)
    l_box = nl*dx

    nx, ny, nz = nl
    print ("nx, ny, nz", nx, ny, nz)
    print ("total data points: ", nx*ny*nz)

    # rr are the coordinates; nn are the director on the coordinate points
    rr = np.zeros ((nx*ny*nz, 3))
    nn = np.zeros ((nx*ny*nz, 3))

    # create a mesh (fill in)
    x = np.linspace(-l_box[0]/2, l_box[0]/2, nx)
    y = np.linspace(-l_box[1]/2, l_box[1]/2, ny)
    z = np.linspace(-l_box[2]/2, l_box[2]/2, nz)
    xv, yv,zv = np.meshgrid(x, y, z,indexing='ij')
    rr[:,0] = xv.flatten();rr[:,1] = yv.flatten();rr[:,2] = zv.flatten()

    # Normalize the directors and avoid 0/0 error
    nn = np.copy(rr)
    nn[:,0] = rr[:,0]/ (np.linalg.norm (rr,axis =1) + 1.0E-5)
    nn[:,1] = rr[:,1]/ (np.linalg.norm (rr,axis =1) + 1.0E-5)
    nn[:,2] = rr[:,2]/ (np.linalg.norm (rr,axis =1) + 1.0E-5)

    # identify which points are outside the sphere and set the director to zero
    idx = np.where (ellip1(rr.T,L, centroid)>0)
    print ("Points that are outside", idx)
    print (len (np.asarray(idx[0])))
    nn[idx] = 0

    # return the parameters
    consts = (l_box,nx,ny,nz,dx)
    return rr, nn, consts


# # Plot midplane to visualize

# In[6]:


def plot_mid(rr,nn,l_box,toplot = False):
    idx2 = np.where (np.abs(rr[:,2]-0)<0.1*np.max(rr) )
    rr2 = rr[idx2]; nn2 = nn[idx2]

    fig, ax = plt.subplots(figsize = (5,5))
    ax.quiver(rr2[:,0], rr2[:,1],  nn2[:,0], nn2[:,1], scale = 50, width = 1.0E-3,color = 'r', pivot = 'mid',headwidth =0, label = 'quintic')

    ax.set_xlabel ("x")
    ax.set_ylabel ("y")

    ax.set_ylim(-l_box[0]/2,l_box[0]/2)
    ax.set_xlim(-l_box[0]/2,l_box[0]/2)

    plt.tight_layout()
    if (toplot):
        plt.savefig("./Images/RadialField.png")
    return


# # Save as a txt file

# In[7]:


def save_txt(rr,nn, consts, fname):
    l_box, nx,ny,nz,dx = consts
    X0 = np.hstack([rr,nn])
    header = 'Radial; director file \n'
    info0 = "Firstline:\nNx"+"\t" + "Ny" +"\t"+ "Nz"+"\t" +"dx"+"\t" + "dy" +"\t"+ "dz"+"\n"
    info1 = "Secondline:\nX_min"+"\t" + "X_max" +"\t"+ "Y_min"+"\t" +"Y_max"+"\t" + "Z_min" +"\t"+ "Z_max"+"\n"
    info2 =  "Third line+ : data\nx \t y \t z \t n_x \t n_y \t n_z \n"
    line0 = np.asarray([nx,ny,nz,dx,dx,dx])
    line1 = np.asarray ([-l_box[0]/2, l_box[0]/2,-l_box[1]/2, l_box[1]/2,-l_box[2]/2, l_box[2]/2])
    X = np.vstack([line0,line1, X0])
    top = header+info0+info1+info2
    np.savetxt(fname, X, fmt='%.4f', delimiter= '\t',header=top)
    return

# In[8]:

# For filenames, replace dot by p
def replace_dot (num):
    pattern = '\.'; repl = 'p'
    string = '%.2f'%num
    res = re.sub(pattern, repl, string, count=0, flags=0)
    return res
# # Run

# In[9]:
if __name__ == "__main__":

    """
    rr, nn, consts = radial (10, dx = 0.1)
    l_box, nx,ny,nz,dx = consts
    plot_mid(rr,nn,l_box,toplot = False)
    fname = "Interpolated_Director_Field/10um-Radial-dx0p1.txt"
    save_txt(rr,nn, consts, fname)
    """
    directory = "Interpolated_Director_Field/"
    Rlist = np.asarray([10.5])
    dxlist = np.asarray([0.2])
    for R in Rlist:
        for dx in dxlist:
            rr, nn, consts = radial (R, dx = dx)
            l_box, nx,ny,nz,dx = consts
            #plot_mid(rr,nn,l_box,toplot = False)
            fname = replace_dot(R) +"um-Radial-"+replace_dot(dx) +".txt"
            print (fname)
            print (directory +fname)
            save_txt(rr,nn, consts, directory+fname)
