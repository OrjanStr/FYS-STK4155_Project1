import numpy as np;
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from frankfunk import Regression

# Loading terrain array
terrain1 = imread('SRTM_data_Norway_2.tif')

resolution = 30 # Meters (according to website)
y_dim, x_dim = len(terrain1[:,0]), len(terrain1[0])

# Setting up x and y arrays
x = np.linspace(0, x_dim-1, x_dim)*resolution
y = np.linspace(0, y_dim-1, y_dim)*resolution
x, y = np.meshgrid(x,y)

# Raveling nata to get into shape (x_dim*y_dim,)
z = terrain1.ravel()
x = x.ravel()
y = y.ravel()

plt.figure()
plt.title('Terrain over Norway')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


reg = Regression()
reg.data_setup(x,y,z)
reg.design_matrix(4)
X_train, X_test, f_train , f_test = reg.split()
f_tilde, f_pred = reg.OLS(X_train, X_test, f_train)

heatmap(x, y, , "m", "m", "Terrain Data")
