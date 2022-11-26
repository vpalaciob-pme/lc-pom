#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import re

# Optional style sheet for beautiful plots
plt.style.use('./large_plot.mplstyle')
#plt.style.use('C:/Users/chenc/Documents/Python_plot_formats/large_plot.mplstyle')
from matplotlib import cm
import matplotlib
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


# In[58]:


def vtk_to_coord(inputfile):

    x = re.split (r"\.", inputfile)
    print ("Frame:", x[2])
    filename2 = "tmp-coordinates.txt"
    file2 = open(filename2, 'w')

    with open(inputfile) as fp:

        nPoints = 0
        file2.write("#Coordinates\n")
        coords = False
        cnt = 0
        for line in fp:
            # convert string to floats
            #
            if (coords == True and cnt < nPoints):
                x = re.split ('\s+', line)

                #print (x)
                for i in range (len(x)):
                    if(re.match ('[0-9]|NaN|-',x[i])):
                        file2.write('%E\t'% float (x[i]))
                file2.write('\n')
                cnt += 1
            if re.match (r'\APOINTS', line):
                print (line)
                x = re.split ('\s+', line)
                nPoints = int (x[1])
                print ("Number of points:", nPoints)
                coords = True

    #print (nPoints)
    return nPoints



# In[60]:


def vtk_to_dir(inputfile, nPoints):

    x = re.split (r"\.", inputfile)
    print ("Frame:", x[2])
    filename2 = "tmp-directors.txt"
    file2 = open(filename2, 'w')

    with open(inputfile) as fp:


        file2.write("#Directors\n")
        coords = False
        cnt = 0
        for line in fp:
            # convert string to floats
            #
            if (coords == True and cnt < (nPoints +2) ):
                x = re.split ('\s+', line)

                #print (x)
                for i in range (len(x)):
                    if(re.match ('[0-9]|NaN|-',x[i])):
                        file2.write('%E\t'% float (x[i]))
                file2.write('\n')
                cnt += 1
            if re.match (r'\Adirector', line):
                print (line)

                print ("Number of points:", nPoints)
                coords = True


    return nPoints



# In[17]:
if __name__ == "__main__":

    directory2 = "./Original_Director_Field/"


    frames= np.loadtxt("tmp-frames.txt", dtype = np.int32)

    for frame in frames:
        fname = "./vtks/stretch."+str(frame)+".vtk"
        #Write temporary files
        nPoints = vtk_to_coord(fname)
        nPoints = vtk_to_dir(fname, nPoints)

        #Write DirectorField file
        coords = np.loadtxt("tmp-coordinates.txt")
        directors = np.loadtxt("tmp-directors.txt")
        X = np.hstack([coords,directors])

        top = "Frame-"+str(frame)+"-Original director file"
        filename1 = directory2+"Frame-"+str(frame) + "-DirectorField.txt"
        np.savetxt(filename1, X, fmt='%.4f', delimiter= '\t',header=top)
