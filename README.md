# LC_POM
Simulate polarized optical microscopy(POM) image from director field. 

## Step 1: create director profile on a regular grid
1. Route 1: 
	- Generate *[ regular-gird director file ]* by ansatz-generate.py
2. Route 2: 
	- Read the director information from .vtk file and write a *[ .csv ]* file by Paraview. 
	- Convert the *[ .csv ]* file to the *[ original director file ]* by director-to-Q.py
	- Interpolate *[ original director file ]* to *[ regular-grid director file ]* by mesh-to-grid.py
2. Route 3: 
	- Output *[ original director file (Q.out) ]* file directly from simulation. 
	- Interpolate *[ original director file ]* to *[ regular-grid director file ]* by mesh-to-grid.py
4. Route 4 (not recommended)
	- Read the director information from .vtk file and convert to *[ original director file ]* by Read_vtk.py. (This function is not robust)
	- Interpolate *[ original director file ]* to *[ regular-grid director file ]* by mesh-to-grid.py
### Read_vtk.py (not recommended)
This function has limited applicability to parse a .vtk file directly to a coordinate file. 
- input: 	.vtk file 
- output:	Frame-#-Coordinates.txt and Frame-#-Directors.txt

### director-to-Q.py
The interpolation of director field from .vtk outputs work better for Q tensor than director field. This file transforms director to Q tensor assuming that Q is locally uniaxial. 

### mesh-to-grid.py 
This function uses interpolation to create director field on a regular grid ready to be read by pom-image.py. 
This file also takes care of rotation by specifiying the Eurler angles. 
	input: 	The Q tensor profile in the folder Original_Director_Field in .csv format (output from Paraview) or .out file where the first block (n_nodes lines) are coordinates and the second block (n_node lines) are the five independent Q tensor entries (q11, q12, q13, q22, q23)
	output: [interpolated-directors].txt such as "radial-16p75-rot-x0y0z0-interpolated-directors.txt"


 *[ regular-grid director file ]* 
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
## Step 2: calculate the POM images

### pom-image.py
Computes the polarized image with various options. 
The script will try to load parameters from params.py. If it doesn't exist, user will be prompted to enter parameters manually.
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
   

