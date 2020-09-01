from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4
z = FrankeFunction(x, y)


# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# Setting up dataset
n = 100
x_ = np.random.rand(n,1)
y_ = np.random.rand(n,1)
x_, y_ = np.meshgrid(x_,y_)
f = FrankeFunction(x_,y_) #+ 0.1*np.random.randn(n,n)

# Setting up design matrix
poly = PolynomialFeatures(4)
X = np.zeros((n,2))
X[:,0] = x_[0]
X[:,1] = y_[:,0]
X = poly.fit_transform(X)

# Splitting into train and test data
X_train, X_test, f_train, f_test = train_test_split(X, f, test_size=0.2)

# Finding coefficients
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ f_train

# Linear Regression
linreg = LinearRegression()
linreg.fit(X_train, f_train)
f_predict = linreg.predict(X)
print(np.shape(f_predict))

ax.scatter(x_,y_,f_predict, s=5)
plt.show()
