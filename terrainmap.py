import numpy as np;
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors
from frankfunc import Regression
from task1 import task_a
from task_b import task_b

# Loading terrain array
terrain1 = imread('SRTM_data_Norway_2.tif')

resolution = 30 # Meters (according to website)
y_dim, x_dim = len(terrain1[:,0]), len(terrain1[0])

# Setting up x and y arrays
x = np.linspace(0, x_dim-1, x_dim)*resolution
y = np.linspace(0, y_dim-1, y_dim)*resolution
x, y = np.meshgrid(x,y) # Using every 50th element

spacing = 1000
# Raveling nata to get into shape (x_dim*y_dim,)
z = terrain1.ravel()[::spacing] # Using every 50th element
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
task_a(x, y, z, generate = False) # Generate=False means we don't generate a new dataset
task_b(x, y, z, data = True)

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

heatmap(x, y, z, "m", "m", "Terrain Data")
