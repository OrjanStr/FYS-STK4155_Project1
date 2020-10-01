import numpy as np;
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors
from frankfunc import Regression, heatmap
import matplotlib.cbook as cbook
import matplotlib.colors as colors

from task1 import task_a
from task_b import task_b
from task_c import task_c
from task_d import task_d
#from task_e import task_e

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


fig, ax = plt.subplots(constrained_layout=True)
colours_sea = plt.cm.terrain(np.linspace(0, 0.17, 256))
colours_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
all_colours = np.vstack((colours_sea, colours_land))
terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map',
    all_colours)

offset = colors.TwoSlopeNorm(vmin=30, vcenter=120, vmax=600)

pcm = ax.pcolormesh(x, y, terrain1, norm = offset,  rasterized=True,
    cmap=terrain_map)
ax.set_xlabel('m')
ax.set_ylabel('m')
ax.set_aspect(1 / np.cos(np.deg2rad(49)))
fig.colorbar(pcm, shrink=0.6, extend='both', label='Elevation')
plt.show()



spacing = 1000
# Raveling nata to get into shape (x_dim*y_dim,)
z = terrain1.ravel()[::spacing]
z = z.astype("float64") # Converting to float
x = x.ravel()[::spacing]
y = y.ravel()[::spacing]

plt.figure()
plt.title('Terrain over Norway')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Looking at MSE and R2 for terraindata
#task_a(x, y, z, generate = False) # Generate=False means we don't generate a new dataset
#task_b(x, y, z, data = True)
#task_c(x, y, z, data = True)
task_d(x, y, z, data = True)
#task_e(x, y, z, data = True)

"""
maxdeg = 10
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
