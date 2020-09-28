import numpy as np;
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from frankfunc import Regression

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
z = z.astype("float64") # Converting to float
x = x.ravel()
y = y.ravel()

z -= np.mean(z)
print("operation successful")

plt.figure()
plt.title('Terrain over Norway')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

maxdeg = 20
reg = Regression()
reg.data_setup(x,y,z)
degrees = np.linspace(1,maxdeg,maxdeg, dtype=int)
train_error = np.zeros(maxdeg)
test_error  = np.zeros(maxdeg)

for i in range(maxdeg):
    deg = degrees[i]
    reg.design_matrix(deg)
    X_train, X_test, f_train , f_test = reg.split(reg.X, reg.f)
    f_tilde, f_pred = reg.OLS(X_train, X_test, f_train)
    train_error[i] = np.mean( (f_tilde - f_train)**2 )
    test_error[i] = np.mean( (f_pred - f_test)**2 )

plt.plot(degrees, train_error, label = 'Train error')
plt.plot(degrees, test_error, label = 'Test error')
plt.legend()
plt.show()

heatmap(x, y, z, "m", "m", "Terrain Data")
