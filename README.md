# LC_POM
Simulate polarized optical microscopy(POM) image from director field.
## Read_vtk.py 
	input: 	.vtk file 
	output:	Frame-#-Coordinates.txt and Frame-#-Directors.txt

## Interpolate_director_field_from_snapshot.py 
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
## cross-pol.py
Computes the polarized image with various options. 
- Input mode: 
	1. single image 
	2. batch processing (specified by file names)
	3. batch processing (specified by frames)
- Polarizer angle: 
	0-180 degrees
- Color mode:
	1. Single wave length
	2. Simplified color
	3. Full spectrum color
   

