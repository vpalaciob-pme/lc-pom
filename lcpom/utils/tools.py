import numpy as np


def normalize(nn):
    for i in range (len (nn)):
        if (np.linalg.norm (nn[i]) >1.0E-5):
            nn[i] = nn[i]/np.linalg.norm (nn[i])
    return nn

def rotate (coords, directors, angles):
    r = R.from_euler('xyz', angles, degrees=True).as_matrix()
    coords = np.matmul (r, coords.T).T
    directors = np.matmul(r, directors.T).T
    return coords, directors


