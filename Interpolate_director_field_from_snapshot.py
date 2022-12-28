#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from matplotlib import cm

# Optional style sheet for beautiful plots
#plt.style.use('C:/Users/chenc/Documents/Python_plot_formats/large_plot.mplstyle')

import numpy.linalg as la

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


# In[2]:


cmap = plt.get_cmap('jet')
plt.style.use('./large_plot.mplstyle')


# In[ ]:


fig, ax = plt.subplots()
for x in np.linspace (0,1,10):
    ax.plot (x, x,color = cmap (x),marker = 'o')


# In[ ]:





# In[4]:


def calc_error_deg (n1, n2):
    if (np.linalg.norm(n1)*np.linalg.norm(n2)>0 ):
        a = (np.dot (n1,n2)/ (np.linalg.norm(n1)*np.linalg.norm(n2)) )
        if (np.abs(a)>1):
            a = a/np.linalg.norm(a)
        res = 180/np.pi*np.arccos(a)
        res = min (180-res, res)

    elif (np.linalg.norm(n1) <1.0E-5 and np.linalg.norm(n1) <1.0E-5):
        res =0
    else:
        res = np.nan
    return res


# In[5]:


def normalize(nn):
    for i in range (len (nn)):
        if (np.linalg.norm (nn[i]) >1.0E-5):
            nn[i] = nn[i]/np.linalg.norm (nn[i])
    return nn


# In[6]:


"""
Method should be quintic, cubit or thin_plate_spline
"""
def interpolate (rr, rr0, nn0, method = 'thin_plate'):

    interp = RBFInterpolator(rr0, nn0, kernel = method, smoothing = 0.1, neighbors = 12)
    nn = interp (rr)
    nn = normalize (nn)
    return nn

def interpolate_s (rr, rr0,  ss0 ,method = 'thin_plate'):

    interp2 = RBFInterpolator(rr0, ss0, kernel = method, smoothing = 0.1, neighbors = 12)
    ss = interp2 (rr)
    return ss


# # Start to interpolate

# In[8]:


def ellip1 (r, L,center):
    x = (r[0]-center[0])*2/L[0]
    y = (r[1]-center[1])*2/L[1]
    z = (r[2]-center[2])*2/L[2]
    return x**2+y**2 +z**2-1


# In[ ]:



# In[9]:


"""
https://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python
"""
pi = np.pi
sin = np.sin
cos = np.cos

def mvee(points, tol = 1.0E-3):
    """
    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1
    """
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = np.dot(u,points)
    A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
               - np.multiply.outer(c,c))/d
    return A, c



# # Find which points correspond to the surface

# In[18]:


def find_surface (coords):
    L0 = np.max(coords, axis=0)-np.min(coords, axis=0)
    centroid0 = np.mean(coords, axis=0)
    idx = np.where (ellip1 (coords.T, L0, centroid0)>-0.08)
    rr= coords[idx]
    surface = np.copy(rr)
    return L0, centroid0, surface


# # Find the smallest enclosing ellipse

# This algorithm flips the axes and returns rx, ry, rz in increasing order.

# In[11]:


def find_L(coords):
    L0, centroid0, surface = find_surface (coords)
    points = surface


    A, centroid = mvee(points, tol = 1.0E-3)
    U, D, V = la.svd(A)
    rx, ry, rz = 1./np.sqrt(D)

    print ("Naive estimation of the long and short axis of the ellipsoid from maximum and minimum of coords.\nDiameters:",L0,"Centroid",centroid0)
    order = np.argsort (L0)
    # Order in the right way
    L1 = np.asarray([2*rx, 2*ry, 2*rz])
    L=np.zeros(3)
    L[order] = L1[:]*0.99

    print("Fitted ellipsoid:")
    print("Diameters:",L,"Centroid",centroid)
    print ("Percentage difference:", (L-L0)/L0*100, "%")
    return L, centroid


# In[12]:


# In[ ]:



def make_grid(L,dx):
    # Box separation grid size: 0.1um
    # Padding: 10%

    l_box = L *1.1
    nl = np.ceil(l_box/dx/10)*10
    nl = np.asarray (nl, dtype= np.int32)
    l_box = nl*dx

    nx, ny, nz = nl
    print ("nx, ny, nz", nx, ny, nz)
    print ("total data points: ", nx*ny*nz)
    if (nx*ny*nz)>1.0E8:
        print ("Grid too fine")
        return
    rr = np.zeros ((nx*ny*nz, 3))
    nn = np.zeros ((nx*ny*nz, 3))

    x = np.linspace(-l_box[0]/2, l_box[0]/2, nx)
    y = np.linspace(-l_box[1]/2, l_box[1]/2, ny)
    z = np.linspace(-l_box[2]/2, l_box[2]/2, nz)
    xv, yv,zv = np.meshgrid(x, y, z,indexing='ij')

    rr[:,0] = xv.flatten();rr[:,1] = yv.flatten();rr[:,2] = zv.flatten()
    rr = np.asarray(rr, dtype = np.float32)
    consts0 =[nx,ny,nz,dx,dx,dx]
    return consts0, l_box, rr, nn



# In[13]:


# In[ ]:





def plot_final (rr, nn, ss, coords, directors, l_box, frame,directory2):
    signs =np.power (-1,directors[:,0]<0)
    directors[:,0] = directors[:,0]*signs
    directors[:,1] = directors[:,1]*signs
    directors[:,2] = directors[:,2]*signs

    idx2 = np.where (np.abs(rr[:,2]-0)<0.025*np.max(coords[:,2]))
    rr2 = rr[idx2]; nn2 = nn[idx2];
    if (ss.any()!= None):
        ss2 = ss[idx2]

    signs =np.power (-1,nn2[:,0]<0)
    nn2[:,0] = nn2[:,0]*signs
    nn2[:,1] = nn2[:,1]*signs
    nn2[:,2] = nn2[:,2]*signs

    idx3 =np.where (np.abs(coords[:,2]-0)<0.2)
    rr0 = coords[idx3]; nn0 = directors[idx3]

    fig, ax = plt.subplots(figsize = (l_box[1],l_box[0]))

    ax.quiver(rr2[:,1], rr2[:,0],  nn2[:,1], nn2[:,0], scale = 50, width = 1.0E-3,color = cmap (np.arctan(nn2[:,1]/(nn2[:,0]+1.0E-3)/np.pi + 1/2)), pivot = 'mid',headwidth =0, label = 'quintic')
    ax.set_xlabel ("x")
    ax.set_ylabel ("y")


    ax.set_xlim(-l_box[1]/2,l_box[1]/2)
    ax.set_ylim(-l_box[0]/2,l_box[0]/2)
    plt.tight_layout()
    fname = directory2 + "Frame-" + str(frame) +"-interp.png"
    plt.savefig(fname)
    ax.quiver(rr0[:,1], rr0[:,0],  nn0[:,1], nn0[:,0], scale = 20, width = 5.0E-3, color = 'k', pivot = 'mid',headwidth =0,  label = 'original')
    fname = directory2 + "Frame-" + str(frame) +"-overlay.png"
    plt.savefig(fname)

    if (ss.any()!= None):
        fig, ax = plt.subplots(figsize = (l_box[1],l_box[0]))

        ax.quiver(rr2[:,1], rr2[:,0],  nn2[:,1], nn2[:,0], scale = 50, width = 1.0E-3,color = cmap (ss2/max(ss2)), pivot = 'mid',headwidth =0, label = 'quintic')
        ax.set_xlabel ("x")
        ax.set_ylabel ("y")
        ax.set_xlim(-l_box[1]/2,l_box[1]/2)
        ax.set_ylim(-l_box[0]/2,l_box[0]/2)
        plt.tight_layout()
        fname = directory2 + "Frame-" + str(frame) +"-interp2.png"
        plt.savefig(fname)

    return


# In[35]:


def write_txt(rr, nn, consts0, l_box,frame,directory2):
    [nx,ny,nz,dx,dy,dz] = consts0
    X0 = np.hstack([rr,nn])
    header = 'Interpolated director file \n'
    info0 = "Firstline:\nNx"+"\t" + "Ny" +"\t"+ "Nz"+"\t" +"dx"+"\t" + "dy" +"\t"+ "dz"+"\n"
    info1 = "Secondline:\nX_min"+"\t" + "X_max" +"\t"+ "Y_min"+"\t" +"Y_max"+"\t" + "Z_min" +"\t"+ "Z_max"+"\n"
    info2 =  "Third line+ : data\nx \t y \t z \t n_x \t n_y \t n_z \n"
    line0 = np.asarray([nx,ny,nz,dx,dy,dz])
    line1 = np.asarray ([-l_box[0]/2, l_box[0]/2,-l_box[1]/2, l_box[1]/2,-l_box[2]/2, l_box[2]/2])

    X = np.vstack([line0,line1, X0])
    top = header+info0+info1+info2
    fname = directory2 + "Frame-"+str(frame)+"-interpolated-directors.txt"
    np.savetxt(fname, X, fmt='%.4f', delimiter= '\t',header=top)

def write_txt_s(rr, nn, ss,consts0, l_box,frame,directory2):
    [nx,ny,nz,dx,dy,dz] = consts0
    X0 = np.hstack([rr,nn,ss])
    header = 'Interpolated director file \n'
    info0 = "Firstline:\nNx"+"\t" + "Ny" +"\t"+ "Nz"+"\t" +"dx"+"\t" + "dy" +"\t"+ "dz"+"\t"+ "max (ss)"+"\n"
    info1 = "Secondline:\nX_min"+"\t" + "X_max" +"\t"+ "Y_min"+"\t" +"Y_max"+"\t" + "Z_min" +"\t"+ "Z_max"+"\t"+ "mean (ss)"+"\n"
    info2 =  "Third line+ : data\nx \t y \t z \t n_x \t n_y \t n_z \t S\n"
    line0 = np.asarray([nx,ny,nz,dx,dy,dz, np.max(ss)])
    line1 = np.asarray ([-l_box[0]/2, l_box[0]/2,-l_box[1]/2, l_box[1]/2,-l_box[2]/2, l_box[2]/2, np.mean(ss[np.where(ss>0)] )])

    X = np.vstack([line0,line1, X0])
    top = header+info0+info1+info2
    fname = directory2 + "Frame-"+str(frame)+"-interpolated-directors.txt"
    np.savetxt(fname, X, fmt='%.4f', delimiter= '\t',header=top)

# In[34]:


"""
The function that takes all the previous functions to read, interpolate and plot
"""
def interp_frame (frame, delta = 0.15,scaling = 1.0, directory1 = "./Original_Director_Field/",directory2 = "./Interpolated_Director_Field/"):

    # Read original director field from directory1
    fname = directory1+"Frame-" + str(frame) +"-DirectorField.txt"
    X = np.loadtxt(fname,dtype = np.float32)
    if (X.shape[1] ==7):
        print ("Has S")
        ss0 = X[:,6]
    else:
        ss0 = None
        ss = None
    coords = X[:,:3]*scaling;directors = X[:,3:6]
    print ("Coords shape",coords.shape)
    print ("Directors shape", directors.shape)
    signs = np.power(-1, (np.sum (np.multiply(directors,coords),1) >0 ))
    #signs =np.power (-1,directors[:,0]<0)
    directors[:,0] = directors[:,0]*signs
    directors[:,1] = directors[:,1]*signs
    directors[:,2] = directors[:,2]*signs

    # Find ellipsoid size
    #L, centroid = find_L(coords)
    L = np.asarray([max(coords[:,0])-min(coords[:,1]),max(coords[:,1])-min(coords[:,0]),max(coords[:,2])-min(coords[:,2])])
    centroid = np.asarray ([np.mean(coords[:,1]),np.mean(coords[:,1]),np.mean(coords[:,2])])
    print ("L:", L, "Centroid:", centroid)

    # Make grid according to size of the ellipsoid
    consts0, l_box, rr, nn = make_grid(L,dx = delta)

    print ("Grid is made. ")
    # Interpolate on actual grid
    nn = interpolate (rr, coords, directors, method = 'thin_plate_spline')
    ss = interpolate_s (rr, coords, ss0, method = 'thin_plate_spline')
    idx = np.where (ellip1(rr.T,L, centroid)>0)
    nn[idx] = 0
    ss[idx] = 0

    # Plot an overlay
    plot_final (rr, nn, ss, coords, directors, l_box, frame, directory2)

    # Save the interpolated director field from directory2
    if (X.shape[1] ==7):
        ss = np.reshape(ss, (len(ss),1))
        write_txt_s(rr, nn,ss, consts0, l_box,frame,directory2)
    else:
        write_txt(rr, nn,ss, consts0, l_box,frame,directory2)

    return


# In[31]:


# In[ ]:




def plot_from_existed(frame,scaling = 1.0,directory1 = "./Original_Director_Field/",directory2 = "./Interpolated_Director_Field/"):

    # Load the interpolated field
    fname = directory2+"Frame-"+str(frame)+"-interpolated-directors.txt"
    X = np.loadtxt(fname,dtype = np.float32);
    [Nx, Ny, Nz] = np.asarray (X[0, :3], dtype = np.int32)
    [dx, dy, dz] = X[0, 3:6]
    [x_min, x_max, y_min, y_max, z_min, z_max] = X[1,0:6]
    print ("Number of data points:", Nx*Ny*Nz)
    print ("dx = %.2f" %(dx))
    # the actual data
    rr = X[2:,:3]; nn = X[2:,3:6]
    fname = directory1+"Frame-" + str(frame) +"-DirectorField.txt"
    X = np.loadtxt(fname,dtype = np.float32)
    coords = X[:,:3]*scaling;directors = X[:,3:6]

    #plot and save
    plot_final (rr, nn,ss, coords, directors, l_box, frame, directory2)

    return

if __name__ == "__main__":

    interp_frame(frame, delta = 0.20,scaling = 21.0/50.0)
    """
    with open("./tmp-filenames.txt") as fp:
            for name in fp:
                #POM_of_Frame(name.strip('\n'), mode,angle)
                POM_of_Frame(name.strip('\n'), mode, angle,wl = np.arange(.400, .681, .014))

    frames= np.loadtxt("tmp-frames.txt", dtype = np.int32)
    for frame in frames:
        print(frame)
        interp_frame(frame)
    """
