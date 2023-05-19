
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from matplotlib import cm
from scipy.spatial.transform import Rotation as R
import numpy.linalg as la
from plum import dispatch

from os import path

# Local imports
from ..utils import *


class LCSystem:

    def __init__(self, coords, director, Sorder):
        """
        Geometry is a class that handles information of the LC system
        coords: coordinates, np.array nnx3
        director: director field, np.array nnx3
        Sorder: scalar order field, np.array nn (optional)
        """
        self.coords = np.array(coords,dtype=float)
        self.director = np.array(director,dtype=float)
        self.Sorder = np.array(Sorder,dtype=float)
    

    
def make_grid(L,dx):
    # Box separation grid size: 0.1um
    # Padding: 10%

    l_box = L *1.1; l_box[2] = L[2]*1.01
    nl = np.ceil(l_box/dx/5)*5
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




def ellip1 (r, L,center):
    x = (r[0]-center[0])*2/L[0]
    y = (r[1]-center[1])*2/L[1]
    z = (r[2]-center[2])*2/L[2]
    return x**2+y**2 +z**2-1





def write_orig(rr, nn, ss,info,euler_angles, directory1):
    ss = ss.reshape ([len(ss),1])
    if (ss.any()!= None):
        X0 = np.hstack([rr,nn,ss])
    else:
        X0 = np.hstack([rr,nn])
    top = "# Euler angles: %.2f, %.2f, %.2f"%(euler_angles[0],euler_angles[1],euler_angles[2])
    fname = directory1 + info +"-original-directors.txt"
    np.savetxt(fname, X0, fmt='%.4f', delimiter= '\t',header=top)
    return




def write_txt(rr, nn, consts0, l_box,info,directory2):
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
    fname = directory2 + info+"-interpolated-directors.txt"
    np.savetxt(fname, X, fmt='%.4f', delimiter= '\t',header=top)
    return


def write_txt_s(rr, nn, ss,consts0, l_box,info,directory2):
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
    fname = directory2 + info +"-interpolated-directors.txt"
    np.savetxt(fname, X, fmt='%.4f', delimiter= '\t',header=top)
    return



"""
The function that takes all the previous functions to interpolate and plot
"""
def interp_frame (system: LCSystem, L , delta, info, directory2 = "./Interpolated_Director_Field/"):

    # Make grid according to size of the ellipsoid
    consts0, l_box, rr, nn = make_grid(L, dx = delta)

    print ("Grid is made. ")
    # Interpolate on actual grid
    centroid = np.asarray ([0,0,0])
    nn = interpolate (rr, system.coords, system.director, method = 'thin_plate_spline')
    idx = np.where (ellip1(rr.T, L, centroid)>0)
    nn[idx] = 0

    if (ss0.any() != None):
        ss = interpolate_s (rr, system.coords, system.Sorder, method = 'thin_plate_spline')
        ss[idx] = 0
    else:
        ss = np.asarray([None])

    # Save the interpolated director field from directory2
    if (ss0.any() != None):
        ss = np.reshape(ss, (len(ss),1))
        write_txt_s(rr, nn,ss, consts0, l_box,info,directory2)
    else:
        write_txt(rr, nn, consts0, l_box,info,directory2)
    
    newsys = LCSystem(rr,nn,ss)

    return newsys



def read_rotate (fname, scaling = 1.0, euler_angles = np.asarray([0,0,0])):

    # Read original director field from directory1

    X = np.loadtxt(fname,dtype = np.float32)
    if (X.shape[1] ==7):
        print ("Has S")
        ss0 = X[:,6]
        print (ss0.shape)
    else:
        ss0 = np.asarray([None])

    coords = X[:,:3]*scaling; directors = X[:,3:6]

    print ("Coords shape",coords.shape)
    print ("Directors shape", directors.shape)


    # Find ellipsoid size
    #L, centroid = find_L(coords)
    L = np.asarray([max(coords[:,0])-min(coords[:,0]),max(coords[:,1])-min(coords[:,1]),max(coords[:,2])-min(coords[:,2])])
    centroid = np.asarray ([np.mean(coords[:,0]),np.mean(coords[:,1]),np.mean(coords[:,2])])
    print ("L:", L, "Centroid:", centroid)

    #R ecenter
    coords[:,0] -= centroid[0]
    coords[:,1] -= centroid[1]
    coords[:,2] -= centroid[2]
    print ("Recentered")

    # Rotate
    if (euler_angles.any()!=0):
        print ("Rotation around x, y, z in order by %.2f, %.2f, %.2f degrees" % (euler_angles[0],euler_angles[1],euler_angles[2]))
        coords, directors = Rotate (coords, directors, euler_angles)

    # Correct signs
    print ("Correct signs of original director")
    signs = np.power(-1, (np.sum (np.multiply(directors,coords),1) >0 ))
    #signs =np.power (-1,directors[:,0]<0)
    directors[:,0] = directors[:,0]*signs
    directors[:,1] = directors[:,1]*signs
    directors[:,2] = directors[:,2]*signs

    return coords, directors, ss0, L





def plot_from_existed(fname_orig, fname_interp, info,scaling = 1.0, euler_angles =  np.asarray ([0,0,0])):

    # the original data
    coords, directors, ss0, L = read_rotate(fname_orig, scaling = scaling, euler_angles = euler_angles )

    # Load the interpolated field
    X = np.loadtxt(fname_interp,dtype = np.float32);
    rr = X[2:,:3]; nn = X[2:,3:6]
    [Nx, Ny, Nz] = np.asarray (X[0, :3], dtype = np.int32)
    [dx, dy, dz] = X[0, 3:6]
    [x_min, x_max, y_min, y_max, z_min, z_max] = X[1,0:6]
    if (X.shape[1] ==7):
        print ("Has S")
        ss = X[2:,6]
    else:
        ss = np.asarray([None])
    print ("Number of data points:", Nx*Ny*Nz)
    print ("dx = %.2f" %(dx))

    #plot and save
    plot_final (rr, nn,ss, coords, directors, L , info, directory2)

    return

