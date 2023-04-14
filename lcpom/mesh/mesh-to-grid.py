
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from matplotlib import cm
from scipy.spatial.transform import Rotation as R
import numpy.linalg as la

from os import path

# Local imports
from ..utils import *

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

cmap = plt.get_cmap('jet')
plt.style.use('./large_plot.mplstyle')

plt.rcParams["figure.dpi"] =50
fig, ax = plt.subplots()
for x in np.linspace (0,1,10):
    ax.plot (x, x,color = cmap (x),marker = 'o')


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



def plot_orig_3views(coords_orig, directors, ss0, L, info,directory2):
    l_box = 1.2*L
    directors_orig = np.copy(directors)
    signs =np.power (-1,directors_orig[:,0]<0)
    directors_orig[:,0] = directors_orig[:,0]*signs
    directors_orig[:,1] = directors_orig[:,1]*signs
    directors_orig[:,2] = directors_orig[:,2]*signs

    # Top
    idx2 = np.where (np.abs(coords_orig[:,2]-0)<0.025*L[2])
    rr0 = coords_orig[idx2]; nn0 = directors_orig[idx2]
    ss2 = ss0[idx2]
    fig, ax = plt.subplots(figsize = (0.5*l_box[0],0.5*l_box[1]))
    #arrowcolors = cmap ((ss2-min(ss2))/(max(ss2)-min(ss2)) );
    arrowcolors = cmap ((ss2-min(ss0))/(max(ss0)-min(ss0)) );
    quiveropts = dict(color=arrowcolors, headlength=0, pivot='middle', scale=2, linewidth=1, units='xy', width=.05, headwidth=1,headaxislength=0)
    ax.quiver(rr0[:,0], rr0[:,1],  nn0[:,0], nn0[:,1], **quiveropts)
    ax.set_xlabel ("x")
    ax.set_ylabel ("y")
    ax.set_xticks ( np.linspace (np.round(-L[0]/2,1) ,np.round(L[0]/2,1), 3 ))
    ax.set_yticks ( np.linspace (np.round(-L[1]/2,1) ,np.round(L[1]/2,1), 3 ))
    ax.set_xlim(np.round(-l_box[0]/2,1), np.round(l_box[0]/2,1) )
    ax.set_ylim(np.round(-l_box[1]/2,1), np.round(l_box[1]/2,1) )
    fig.set_size_inches(0.5*l_box[0],0.5*l_box[1])
    plt.tight_layout()
    fname = directory2 + info +"-orig-topview.png"
    plt.savefig(fname)

    # x
    idx2 = np.where (np.abs(coords_orig[:,0]-0)<0.025*L[0])
    rr0 = coords_orig[idx2]; nn0 = directors_orig[idx2]
    ss2 = ss0[idx2]
    fig, ax = plt.subplots(figsize = (0.5*l_box[1],0.5*l_box[2]))
    #arrowcolors = cmap ((ss2-min(ss2))/(max(ss2)-min(ss2)) );
    arrowcolors = cmap ((ss2-min(ss0))/(max(ss0)-min(ss0)) );
    quiveropts = dict(color=arrowcolors, headlength=0, pivot='middle', scale=2, linewidth=1, units='xy', width=.05, headwidth=1,headaxislength=0)
    ax.quiver(rr0[:,1], rr0[:,2],  nn0[:,1], nn0[:,2], **quiveropts)
    ax.set_xlabel ("y")
    ax.set_ylabel ("z")
    ax.set_xticks ( np.linspace (np.round(-L[1]/2,1) ,np.round(L[1]/2,1), 3 ))
    ax.set_yticks ( np.linspace (np.round(-L[2]/2,1) ,np.round(L[2]/2,1), 3 ))
    ax.set_xlim(np.round(-l_box[1]/2,1), np.round(l_box[1]/2,1) )
    ax.set_ylim(np.round(-l_box[2]/2,1), np.round(l_box[2]/2,1) )
    fig.set_size_inches(0.5*l_box[1],0.5*l_box[2])
    plt.tight_layout()
    fname = directory2 + info +"-orig-frontview.png"
    plt.savefig(fname)

    idx2 = np.where (np.abs(coords_orig[:,1]-0)<0.025*L[1])
    rr0 = coords_orig[idx2]; nn0 = directors_orig[idx2]
    ss2 = ss0[idx2]
    fig, ax = plt.subplots(figsize = (0.5*l_box[0],0.5*l_box[2]))
    #arrowcolors = cmap ((ss2-min(ss2))/(max(ss2)-min(ss2)) );
    arrowcolors = cmap ((ss2-min(ss0))/(max(ss0)-min(ss0)) );
    quiveropts = dict(color=arrowcolors, headlength=0, pivot='middle', scale=2, linewidth=1, units='xy', width=.05, headwidth=1,headaxislength=0)
    ax.quiver(rr0[:,0], rr0[:,2],  nn0[:,0], nn0[:,2], **quiveropts)
    ax.set_xlabel ("x")
    ax.set_ylabel ("z")
    ax.set_xticks ( np.linspace (np.round(-L[0]/2,1) ,np.round(L[0]/2,1), 3 ))
    ax.set_yticks ( np.linspace (np.round(-L[2]/2,1) ,np.round(L[2]/2,1), 3 ))
    ax.set_xlim(np.round(-l_box[0]/2,1), np.round(l_box[0]/2,1) )
    ax.set_ylim(np.round(-l_box[2]/2,1), np.round(l_box[2]/2,1) )
    fig.set_size_inches(0.5*l_box[0],0.5*l_box[2])
    plt.tight_layout()
    fname = directory2 + info +"-orig-leftview.png"
    plt.savefig(fname)



"""https://stackoverflow.com/questions/37154071/python-quiver-plot-without-head"""
def plot_orig (coords_orig, directors, ss0, L, info,directory2):
    #signs =np.power (-1,directors_orig[:,0]<0)
    l_box = 1.2*L
    directors_orig = np.copy(directors)
    signs =np.power (-1,directors_orig[:,0]<0)
    directors_orig[:,0] = directors_orig[:,0]*signs
    directors_orig[:,1] = directors_orig[:,1]*signs
    directors_orig[:,2] = directors_orig[:,2]*signs

    # z-cross-section
    idx2 = np.where (np.abs(coords_orig[:,2]-0)<0.025*L[2])
    rr0 = coords_orig[idx2]; nn0 = directors_orig[idx2]

    fig, ax = plt.subplots(figsize = (12,12))
    arrowcolors = cmap (np.arctan(nn0[:,1]/(nn0[:,0]+1.0E-3)/np.pi + 1/2))
    #alpha = np.arctan(nn0[:,1]/(nn0[:,0]+1.0E-3))
    #arrowcolors = cmap (np.sin(2*np.abs (alpha)))
    quiveropts = dict(color=arrowcolors, headlength=0, pivot='middle', scale=2, linewidth=1, units='xy', width=.05, headwidth=1,headaxislength=0)

    #quiveropts = scale = 10, width = 1.0E-2,color = cmap (np.arctan(nn0[:,1]/(nn0[:,0]+1.0E-3)/np.pi + 1/2)), pivot = 'mid',headwidth =0
    ax.quiver(rr0[:,0], rr0[:,1],  nn0[:,0], nn0[:,1], **quiveropts)

    ax.set_xticks ( np.linspace (np.round(-L[0]/2,1) ,np.round(L[0]/2,1), 3 ))
    ax.set_yticks ( np.linspace (np.round(-L[1]/2,1) ,np.round(L[1]/2,1), 3 ))
    ax.set_xlim(np.round(-l_box[0]/2,1), np.round(l_box[0]/2,1) )
    ax.set_ylim(np.round(-l_box[1]/2,1), np.round(l_box[1]/2,1) )
    ax.set_xlabel ("x")
    ax.set_ylabel ("y")
    dpi = matplotlib.rcParams['savefig.dpi']
    fig.set_size_inches(0.5*l_box[0],0.5*l_box[1])

    plt.tight_layout()
    fname = directory2 + info +"-orig1.png"
    plt.savefig(fname)



    if (ss0.any()!= None):
        plot_orig_3views(coords_orig, directors, ss0, L, info,directory2)

    return




def plot_final (rr, nn, ss, coords, directors, L, info, directory2):

    directors_orig = np.copy(directors)
    print ("Plotting droplet with size:", L)
    l_box = 1.2*L
    signs =np.power (-1,directors_orig[:,0]<0)
    directors_orig[:,0] = directors_orig[:,0]*signs
    directors_orig[:,1] = directors_orig[:,1]*signs
    directors_orig[:,2] = directors_orig[:,2]*signs

    idx2 = np.where (np.abs(rr[:,2]-0)<0.025*L[2])
    rr2 = rr[idx2]; nn2 = nn[idx2];
    if (ss.any()!= None):
        ss2 = ss[idx2]

    signs =np.power (-1,nn2[:,0]<0)
    nn2[:,0] = nn2[:,0]*signs
    nn2[:,1] = nn2[:,1]*signs
    nn2[:,2] = nn2[:,2]*signs

    idx3 =np.where (np.abs(coords[:,2]-0)<0.025*L[2])
    rr0 = coords[idx3]; nn0 = directors_orig[idx3]

    fig, ax = plt.subplots(figsize = (10,10))
    arrowcolors = cmap (np.arctan(nn2[:,1]/(nn2[:,0]+1.0E-3)/np.pi + 1/2))
    quiveropts = dict(color=arrowcolors, headlength=0, pivot='middle', scale=2, linewidth=1, units='xy', width=.05, headwidth=1,headaxislength=0)
    ax.quiver(rr2[:,0], rr2[:,1],  nn2[:,0], nn2[:,1], **quiveropts)
    ax.set_xlabel ("x")
    ax.set_ylabel ("y")
    ax.set_xticks ( np.linspace (np.round(-L[0]/2,1) ,np.round(L[0]/2,1), 3 ))
    ax.set_yticks ( np.linspace (np.round(-L[1]/2,1) ,np.round(L[1]/2,1), 3 ))
    ax.set_xlim(np.round(-l_box[0]/2,1), np.round(l_box[0]/2,1) )
    ax.set_ylim(np.round(-l_box[1]/2,1), np.round(l_box[1]/2,1) )
    fig.set_size_inches(0.5*l_box[0],0.5*l_box[1])
    plt.tight_layout()
    fname = directory2 + info +"-interp.png"
    plt.savefig(fname)

    quiveropts = dict(color='k', headlength=0, pivot='middle', scale=2, linewidth=1, units='xy', width=.05, headwidth=1,headaxislength=0)
    ax.quiver(rr0[:,0], rr0[:,1],  nn0[:,0], nn0[:,1], **quiveropts)
    fname = directory2 + info +"-overlay.png"
    plt.savefig(fname)

    if (ss.any()!= None):


        fig, ax = plt.subplots(figsize = (10,10))
        arrowcolors = cmap ((ss2-min(ss2))/(max(ss2)-min(ss2)) );
        quiveropts = dict(color=arrowcolors, headlength=0, pivot='middle', scale=2, linewidth=1, units='xy', width=.05, headwidth=1,headaxislength=0)
        ax.quiver(rr2[:,0], rr2[:,1],  nn2[:,0], nn2[:,1], **quiveropts)
        ax.set_xlabel ("x")
        ax.set_ylabel ("y")
        ax.set_xticks ( np.linspace (np.round(-L[0]/2,1) ,np.round(L[0]/2,1), 3 ))
        ax.set_yticks ( np.linspace (np.round(-L[1]/2,1) ,np.round(L[1]/2,1), 3 ))
        ax.set_xlim(np.round(-l_box[0]/2,1), np.round(l_box[0]/2,1) )
        ax.set_ylim(np.round(-l_box[1]/2,1), np.round(l_box[1]/2,1) )
        fig.set_size_inches(0.5*l_box[0],0.5*l_box[1])
        plt.tight_layout()

        fname = directory2 + info +"-interp2.png"
        plt.savefig(fname)

    del (directors_orig)
    return



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
def interp_frame (coords, directors, ss0, L , delta, info, directory2 = "./Interpolated_Director_Field/"):

    # Make grid according to size of the ellipsoid
    consts0, l_box, rr, nn = make_grid(L, dx = delta)

    print ("Grid is made. ")
    # Interpolate on actual grid
    centroid = np.asarray ([0,0,0])
    nn = interpolate (rr, coords, directors, method = 'thin_plate_spline')
    idx = np.where (ellip1(rr.T, L, centroid)>0)
    nn[idx] = 0

    if (ss0.any() != None):
        ss = interpolate_s (rr, coords, ss0, method = 'thin_plate_spline')
        ss[idx] = 0
    else:
        ss = np.asarray([None])

    # Plot an overlay
    plot_final (rr, nn, ss, coords, directors, L,info, directory2)

    # Save the interpolated director field from directory2
    if (ss0.any() != None):
        ss = np.reshape(ss, (len(ss),1))
        write_txt_s(rr, nn,ss, consts0, l_box,info,directory2)
    else:
        write_txt(rr, nn, consts0, l_box,info,directory2)


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





def batch(dx=0.2, loadDirectory = "./Original_Director_Field/", saveDirectory = "./Interpolated_Director_Field/"):
    with open("./interp-filenames.txt") as fp:
        for fname in fp:
            #POM_of_Frame(name.strip('\n'), mode,angle)
            info = path.splitext(fname)[0]
            fpath = loadDirectory+fname
            myangles = np.asarray([0,0,0])
            print (fpath)
            print("info:", info)
            info = "rot-x"+str(myangles[0])+"y"+str(myangles[1])+"z"+str(myangles[2])
            coords, directors, ss0, L = read_rotate(fpath, scaling = 1.0, euler_angles = myangles)
            plot_orig (coords, directors, ss0, L, info, saveDirectory)
            write_orig(coords, directors, ss0,info,euler_angles = myangles, directory1 = loadDirectory)
            interp_frame (coords, directors, ss0, L , delta = dx, info = info, directory2 = saveDirectory)
        return



if __name__ == "__main__":
    batch(dx =0.2)




"""
info = "top-down"
directory1 = "./Original_Director_Field/"
fname_orig = directory1+info+"-original-directors.txt"
#fname_orig = directory1+"Real-Radial"+".txt"

directory2 = "./Interpolated_Director_Field/"
fname_interp = directory2+info+"-interpolated-directors.txt"
plot_from_existed(fname_orig, fname_interp,info,scaling = 1.0,euler_angles= np.asarray ([0,0,0]))
""";
