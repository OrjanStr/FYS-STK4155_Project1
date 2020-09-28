import numpy as np;
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Loading terrain array
terrain1 = imread('SRTM_data_Norway_2.tif')

resolution = 30 # Meters (according to website)
y_dim, x_dim = len(terrain1[:,0]), len(terrain1[0])

# Setting up x and y arrays
x = np.linspace(0, x_dim, x_dim-1)*resolution
y = np.linspace(0, y_dim, y_dim-1)*resolution

plt.figure()
plt.title('Terrain over Norway')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
