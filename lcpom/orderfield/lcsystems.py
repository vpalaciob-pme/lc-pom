
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
from ..utils.tools import *

class Grid:
    def __init__(
        self,
        Lbox,
        nx, ny, nz,
        deltax,
        grid_coords
    ):
        self.Lbox = Lbox
        self.nl = np.asarray([nx, ny, nz])
        self.nn = nx*ny*nz
        self.dx = np.asarray([deltax, deltax, deltax])
        self.xyz = grid_coords
  

class OrderField:
    @dispatch
    def __init__(
        self,
        coords: np.ndarray,
        S : float,
        director: np.ndarray
    ):
        self.coords = coords
        self.director = director
        self.scalar = np.ones((director.size(),1),dtype=float)*S
        self.Qtensor = self.calculate_tensor(S,director)

    @dispatch
    def __init__(
        self,
        coords: np.ndarray,
        S : np.ndarray,
        director: np.ndarray
    ):
        self.coords = coords
        self.director = director
        self.scalar = S
        self.Qtensor = self.calculate_tensor(S,director)

    @dispatch
    def __init__(
        self,
        coords: np.ndarray,
        Qorder: np.ndarray
    ):
        self.coords = coords
        self.Qtensor = Qorder
        self.scalar, self.director = self.calculate_scalar_vector(Qorder)
       
    def calculate_scalar_vector(Qorder: np.ndarray):
        nodes = Qorder.size()[0]
        n = np.zeros((nodes,3))
        S = np.zeros((nodes,1))

        for i in range(0,nodes-1):
            Q = self.v_to_m(Qorder[i,:])
            vals, vects = la.eig(Q)
            j = vals.argmax()
            S[i] = vals[j]*1.5
            n[i] = vects[:,j]

        return S, n

    def calculate_tensor(S: np.ndarray, n: np.ndarray):
        nodes = n.size()[0]
        QQ = np.zeros((nodes,5))

        for i in range(0,nodes-1):
            nn = n[i,:]
            SS = S[i]
            QQ[i,0] = SS*(n[0]*n[0] - 1./3.)
            QQ[i,1] = SS*n[0]*n[1]
            QQ[i,2] = SS*n[0]*n[2]
            QQ[i,3] = SS*(n[1]*n[1] - 1./3.)
            QQ[i,4] = SS*n[1]*n[2]           

        return QQ
    
    def v_to_m(v: np.ndarray):
        """
        Converts a vector of size 5 into a symmetric and traceless 3x3 matrix
        Useful when converting the independent entries of Q onto the tensor form
        """
        m = np.zeros((3,3))

        m[0][0] = v[0]
        m[0][1] = v[1]
        m[0][2] = v[2]
        m[1][0] = v[1]
        m[1][1] = v[3]
        m[1][2] = v[4]
        m[2][0] = v[2]
        m[2][1] = v[4]
        m[2][2] = - v[0] - v[3]

        return m
        

class LCSystem:
    @dispatch
    def __init__(
        self,
        order: OrderField,
        material: str = "5CB"
    ):
        """
        LCSystem is a class that handles information of the LC system
        coords: coordinates, np.array nnx3
        director: director field, np.array nnx3
        Sorder: scalar order field, np.array nn (optional)
        """
        self.all_order = order
        self.coords = order.coords
        self.Sorder = order.scalar
        self.director = order.director
        self.material = material

    def calculate_box(self):
        """
        Calculates the enclosing box of the system in 3 dimensions.
        Returns a 3 vector with the length of the box in x, y, z.
        """
        self.L = np.asarray([0,0,0],dtype=float)
        centroid = np.mean(self.coords,axis=1)
        self.coords -= centroid                 # Center the system at the origin.
        Lmin = np.min(self.coords)
        Lmax = np.max(self.coords)
        self.L = Lmax - Lmin


class LCGrid:

    @dispatch
    def __init__(
            self,
            grid: Grid,
            order: OrderField,
            material: str = "5CB"
    ):
        """
        LCGrid is a class that handles the LC information once scalar and director order fields are interpolated onto grid
    
        """
        self.grid = grid
        self.Sorder = order.scalar
        self.director = order.director
        self.material = material
        self.no : np.ndarray
        self.ne : np.ndarray  
     
    @dispatch
    def calculate_n(self, lamb: float):
        """
        Calculates the refractive indices (n_o, n_e) of 5CB for the wavelength lamb
        """
        l1 = 0.210; l2 = 0.282
        n0e = 0.455; g1e = 2.325; g2e = 1.397
        n0o = 0.414; g1o = 1.352; g2o = 0.470

        self.ne = 1 + n0e + g1e*(lamb**2 * l1**2)/(lamb**2-l1**2) + g2e*(lamb**2 * l2**2)/(lamb**2-l2**2)
        self.no = 1 + n0o + g1o*(lamb**2 * l1**2)/(lamb**2-l1**2) + g2o*(lamb**2 * l2**2)/(lamb**2-l2**2)


    @dispatch
    def calculate_n(self, lamb: float, s: np.ndarray):
        """
        Calculates the refractive indices (n_o,n_e) of 5CB for the wavelength lamb
        and order parameter s
        """
        l1 = 0.210
        l2 = 0.282
        n0e = 0.455; g1e = 2.325; g2e = 1.397
        n0o = 0.414; g1o = 1.352; g2o = 0.470

        n_e = 1 + n0e + g1e*(lamb**2 * l1**2)/(lamb**2-l1**2) + g2e*(lamb**2 * l2**2)/(lamb**2-l2**2)
        n_o = 1 + n0o + g1o*(lamb**2 * l1**2)/(lamb**2-l1**2) + g2o*(lamb**2 * l2**2)/(lamb**2-l2**2)

        S0 = 0.68
        delta_n = (n_e - n_o)/S0
        abt = (n_e + 2*n_o)/3.0
        self.ne = abt + 2/3*s*delta_n
        self.no = abt - 1/3*s*delta_n
        

def make_grid(L: np.ndarray, dx: np.ndarray ):
    """
    Discretizes a box of size (L,L,L) into a regular grid with 100nm = 0.1um spacing between points.
    """
    # Box separation grid size: 0.1um
    # Padding: 10%

    Lbox = L *1.1
    Lbox[2] = L[2]*1.01
    nl = np.ceil(Lbox/dx[0]/5)*5   ### Ask Elise: Why 5?
    nl = np.asarray (nl, dtype= np.int32)
    Lbox = nl*dx

    nx, ny, nz = nl
    nn = nx*ny*nz
    
    print ("nx, ny, nz", nx, ny, nz)
    print ("total data points: ", nn)
    if ( nn )>1.0E8:
        print ("Resolution of the grid is too high. Please coarsen.")
        return
    rr = np.zeros ( (nn, 3) )
    
    x = np.linspace(-Lbox[0]/2, Lbox[0]/2, nx)
    y = np.linspace(-Lbox[1]/2, Lbox[1]/2, ny)
    z = np.linspace(-Lbox[2]/2, Lbox[2]/2, nz)
    xv, yv, zv = np.meshgrid(x, y, z,indexing='ij')

    rr[:,0] = xv.flatten();rr[:,1] = yv.flatten();rr[:,2] = zv.flatten()
    xyz = np.asarray(rr, dtype = np.float32)
    
    grid = Grid( Lbox, nx, ny, nz, dx, xyz)

    return grid


def interp_frame (system: LCSystem, delta: float = 0.1 ):

    """
    Interp_frame interpolates the order field data onto a grid with finer resolution
    """

    # Make grid according to size of the ellipsoid
    grid = make_grid(system.L, dx = delta)

    # Interpolate data onto finer grid
    centroid = np.mean (system.coords,axis=1)
    nn = interpolate (grid.xyz, system.coords, system.director, method = 'thin_plate_spline')
    nn = normalize_vector(nn)
    idx = np.where (ellip1(grid.xyz.T, system.L_box, centroid)>0)    # Finds nodes where LC does not exist
    nn[idx] = 0


    ss = interpolate (grid.xyz, system.coords, system.Sorder, method = 'thin_plate_spline')
    ss[idx] = 0

    # Save the interpolated director field from directory2
    #if ( system.Sorder.any() != None):
    #    ss = np.reshape(ss, (len(ss),1))
    #    write_txt_s(rr, nn,ss, consts0, l_box, info, directory2)
    #else:
    #    write_txt(rr, nn, consts0, l_box,info,directory2)
    
    LCinfo = LCGrid(grid, ss, nn, "5CB")

    return LCinfo


def interpolate (x0, x1, y0, method = 'thin_plate'):

    interp = RBFInterpolator(x0, y0, kernel = method, smoothing = 0.1, neighbors = 12)
    y1 = interp (x1)

    return y1

### Old functions that I'm not sure are needed.
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

