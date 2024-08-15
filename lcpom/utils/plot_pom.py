
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
