from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class Regression:
    def __init__(self,n,deg):
        self.n = n
        self.deg = deg

    def FrankeFunction(self,x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
        return term1 + term2 + term3 + term4

    def plot_franke(self):
        # Make data.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x = np.arange(0, 1, 0.05)
        y = np.arange(0, 1, 0.05)
        x, y = np.meshgrid(x,y)
        z = self.FrankeFunction(x, y)

        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

    def dataset2D(self):
        # Setting up dataset
        x_ = np.random.rand(self.n, 1)
        y_ = np.random.rand(self.n, 1)
        self.x_, self.y_ = np.meshgrid(x_, y_)
        self.f = self.FrankeFunction(self.x_, self.y_) + 0.1*np.random.randn(self.n, self.n)

    def design_matrix(self):
        # Setting up design matrix
        poly = PolynomialFeatures(self.deg)
        X = np.zeros((self.n,2))
        X[:,0] = self.x_[0]
        X[:,1] = self.y_[:,0]
        self.X = poly.fit_transform(X)

    def linear_regression(self, ts):
        # Splitting into train and test data
        X_train, X_test, f_train, f_test = train_test_split(self.X, self.f, test_size=ts)
        # Linear Regression
        linreg = LinearRegression()
        linreg.fit(X_train, f_train)
        self.f_predict = linreg.predict(self.X)
        return self.f_predict

    def betas(self,X,y):
        # Finding coefficients
        beta = np.linalg.inv(X.T @ X) @ X.T @ f
        return beta

reg = Regression(100,2)
reg.dataset2D()
reg.design_matrix()
f_pred = reg.linear_regression(ts = 0.2)
