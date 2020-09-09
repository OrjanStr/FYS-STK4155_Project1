from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

class Regression:
    def __init__(self,n):
        self.n = n



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
        self.x_ = np.random.randn(self.n)
        self.y_ = np.random.randn(self.n)

        # Setting up the FrankeFunction with added noise
        noise = 0.1*np.random.randn(self.n)

        self.f = self.FrankeFunction(self.x_, self.y_) + noise

        # Calculating variance of noise for later use
        self.sigma_squared = 1.0/self.n * np.sum( noise**2 )

    def design_matrix(self):
        # Setting up design matrix
        poly = PolynomialFeatures(self.deg)
        X = np.zeros((self.n,2))
        X[:,0] = self.x_[0]
        X[:,1] = self.y_[:,0]
        self.X = poly.fit_transform(X)

    def linear_regression(self, ts=0.25):
        # Splitting into train and test data
        self.X_train, self.X_test, self.f_train, self.f_test = train_test_split(self.X, self.f, test_size=ts)
        # Linear Regression

        linreg = LinearRegression()
        linreg.fit(self.X_train, self.f_train)
        self.f_predict = linreg.predict(self.X)

        self.y_train_pred = linreg.predict(self.X_train)
        self.y_test_pred = linreg.predict(self.X_test)

        return self.f_predict

    def design_matrix_homemade(self, deg):
        # Setting up matrix


        self.p = int(0.5 * (deg + 1) * (deg + 2))
        X = np.zeros((self.n, self.p))

        # Filling in values
        idx = 0
        for i in range(deg+1):
            for j in range(deg+1-i):
                X[:,idx] = self.x_**i * self.y_**j
                idx += 1

        self.X = X


    def betas(self,X,z):
        # Finding coefficients
        beta = np.linalg.pinv(X.T @ X) @ X.T @ z
        return beta

    def linear_regression_homemade(self, ts=0.20):
        # Splitting into train and test data
        self.X_train, self.X_test, self.f_train, self.f_test = train_test_split(self.X, self.f, test_size=ts)
        self.B = self.betas(self.X_train, self.f_train)
        # Setting up model
        self.y_train_pred = self.X_train @ self.B
        self.y_test_pred = self.X_test @ self. B
        return self.y_test_pred ,self.y_train_pred

    def mean_squared_error_homemade(self, y, y_tilde):
        self.MSE = 1.0/self.n * np.sum((y - y_tilde)**2)
        return self.MSE

    def r_squared(self, y, y_tilde):
        y_mean = 1.0/self.n * np.sum(y)
        top = np.sum( (y - y_tilde)**2 )
        bottom = np.sum( (y - y_mean)**2 )
        self.R2 = 1 - top/bottom
        return self.R2

    def bias_variance_plot(self):

        max_complexity = 12
        trials = 100
        complexity = np.linspace(1,max_complexity,max_complexity)
        test_err = np.zeros(len(complexity))
        train_err = np.zeros(len(complexity))


        for deg in range(1,max_complexity):
            test_err[deg] = 0
            train_err[deg] = 0

            for samples in range (trials):
                self.design_matrix_homemade(deg)
                self.linear_regression_homemade()
                #self.design_matrix()
                #self.linear_regression()


            test_err[deg] += mean_squared_error(self.f_test,self.y_test_pred)
            train_err[deg] += mean_squared_error(self.f_train,self.y_train_pred)


            test_err[deg] /= trials
            train_err[deg] /= trials



        plt.plot(complexity, np.log10(train_err), label='Training Error')
        plt.plot(complexity, np.log10(test_err), label='Test Error')
        plt.xlabel('Polynomial degree')
        plt.ylabel('log10[MSE]')
        plt.legend()
        plt.show()


reg = Regression(100)
reg.dataset2D()#mse_train[i] = self.mean_squared_error(y_model[:75],self.f_train)
            #mse_test[i] = self.mean_squared_error(y_model[:25],self.f_test)
reg.design_matrix_homemade(4)
f_pred, f_tilde = reg.linear_regression_homemade(0.2)
print("R2: ", reg.r_squared(reg.f_train, f_tilde))
print("MSE: ", reg.mean_squared_error_homemade(reg.f_test, f_pred))

print(np.shape(reg.f))
reg.bias_variance_plot()
