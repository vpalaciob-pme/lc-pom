# LC_POM
Simulate polarized optical microscopy(POM) image from director field.
## Read_vtk.ipynb 
	input: 	.vtk file 
	output:	Frame-#-Coordinates.txt and Frame-#-Directors.txt

## Interpolate...from-snapshot.ipynb 
	input: 	Frame-#-Coordinates.txt and Frame-#-Directors.txt
	output: 	interpolated-directors.txt

To read interpolated-directors.txt: 

fname = "interpolated-directors.txt"
X = np.loadtxt(fname,dtype = np.float32);
```
  # the header two lines
  [Nx, Ny, Nz] = np.asarray (X[0, :3], dtype = np.int32) 
  [dx, dy, dz] = X[0, 3:] 
  [x_min, x_max, y_min, y_max, z_min, z_max] = X[1] 
  # the actual data 
  rr = X[2:,:3]; nn = X[2:,3:] 
```
