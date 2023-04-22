import numpy as np
import matplotlib.plot as plt


def rotate (coords, directors, angles):
    r = R.from_euler('xyz', angles, degrees=True).as_matrix()
    coords = np.matmul (r, coords.T).T
    directors = np.matmul(r, directors.T).T
    return coords, directors

def plot_image (intensity, vmax = None,savename = None):
    fig, ax = plt.subplots()
    if (len(intensity.shape) == 3):
        image = np.transpose(intensity, [1,0,2])
    else:
        image = np.transpose(intensity)
    if (vmax == None):
        vmax = np.max(image)
    im = ax.imshow(image, cmap=plt.get_cmap('bone'),interpolation='bicubic',origin = 'lower', vmax = vmax)
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
