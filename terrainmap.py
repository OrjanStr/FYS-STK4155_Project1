import numpy as np;
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors
from linear_regression import Regression
import matplotlib.cbook as cbook
import matplotlib.colors as colors

from task_a import task_a
from task_b import task_b
from task_c import task_c
from task_d import task_d
from task_e import task_e

# Loading terrain array
terrain1 = imread('SRTM_data_Norway_2.tif')
terrain2 = np.asarray(terrain1)

print(terrain1.shape)

resolution = 10 # Meters (according to website)
y_dim, x_dim = len(terrain1[:,0]), len(terrain1[0])

# Setting up x and y arrays
x = np.linspace(0, x_dim-1, x_dim)*resolution
y = np.linspace(0, y_dim-1, y_dim)*resolution
x, y = np.meshgrid(x,y)

#colour plot for terrain data
# fig, ax = plt.subplots(constrained_layout=True)
# colours_sea = plt.cm.terrain(np.linspace(0, 0.17, 256))
# colours_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
# all_colours = np.vstack((colours_sea, colours_land))
# terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map',
#     all_colours)

# offset = colors.TwoSlopeNorm(vmin=20, vcenter=80, vmax=600)

# pcm = ax.pcolormesh(x, y, terrain1, norm = offset,  rasterized=True,
#     cmap=terrain_map, shading= 'auto')
# ax.set_xlabel('m',fontsize='16')
# ax.set_ylabel('m',fontsize='16')
# ax.set_aspect(1 / np.cos(np.deg2rad(20)))
# fig.colorbar(pcm, shrink=0.6, extend='both', label='Elevation')
# plt.title('Terrain Data',fontsize='16')

# plt.savefig('visuals/terrain.pdf')
# plt.show()

# Converting Terraindata for calculations
spacing = 1000
# Raveling nata to get into shape (x_dim*y_dim,)
z = terrain1.ravel()[::spacing]
z = z.astype("float64") # Converting to float
x = x.ravel()[::spacing]
y = y.ravel()[::spacing]

#black and white plot for terrain data
# plt.figure()
# plt.title('Terrain over Norway')
# plt.imshow(terrain1, cmap='gray')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

# Looking at MSE and R2 for terraindata
#task_a(x, y, z, generate = False) # Generate=False means we don't generate a new dataset
maxdeg = 10
# task_b(maxdeg, x, y, z, data = True)
# task_c(maxdeg, x, y, z, data = True)
lam_lst = np.logspace(-15,-8,20)
maxdeg = 10
# task_d(maxdeg, lam_lst, x, y, z, data = True)
task_e(maxdeg, lam_lst, 50, x, y, z, data = True)

"""
maxdeg = 30
reg = Regression()
reg.data_setup(x,y,z)
degrees = np.linspace(1,maxdeg,maxdeg, dtype=int)

# OLS arrays ---------
train_error_OLS = np.zeros(maxdeg)
test_error_OLS  = np.zeros(maxdeg)
# MSE, Bias and Variance
bias_OLS = np.zeros(maxdeg)
bias_OLS = np.zeros(maxdeg)

for i in range(maxdeg):
    deg = degrees[i]
    reg.design_matrix(deg)
    X_train, X_test, f_train , f_test = reg.split(reg.X, reg.f)

    f_tilde, f_pred = reg.OLS(X_train, X_test, f_train)
    train_error_OLS[i] = np.mean( (f_tilde - f_train)**2 )
    test_error_OLS[i] = np.mean( (f_pred - f_test)**2 )

plt.plot(degrees, train_error_OLS, label = 'Train error')
plt.plot(degrees, test_error_OLS, label = 'Test error')
plt.legend()
plt.savefig("visuals/terrainplot.pdf")
plt.show()


print ('shape f_pred: ' ,f_pred.shape, 'y shape: ', y.shape)

x_plot= np.linspace(0, x[-1], 24)
y_plot= np.linspace(0,y[-1], 54)
f_plot = np.reshape(f_pred[:1296], (54,24))
heatmap(x_plot, y_plot, f_plot, "m", "m", "Terrain Data")
"""
