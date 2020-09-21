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
from sklearn.utils import resample



class Regression:

    def __init__(self,n):

        self.n = n

    def _franke_function(self,x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

        return term1 + term2 + term3 + term4

    def dataset2D(self):
        # Setting up dataset
        self.x = np.random.randn(self.n)
        self.y = np.random.randn(self.n)

        # Setting up the FrankeFunction with added noise
        noise = 0.1*np.random.randn(self.n)

        self.f = self._franke_function(self.x, self.y) + noise

        # Calculating variance of noise for later use
        self.sigma_squared = 1.0/self.n * np.sum( noise**2 )

    def design_matrix(self, deg):
        # Setting up matrix
        self.p = int(0.5 * (deg + 1) * (deg + 2))
        X = np.zeros((self.n, self.p))
        # Filling in values
        idx = 0
        for i in range(deg+1):
            for j in range(deg+1-i):
                X[:,idx] = self.x**i * self.y**j
                idx += 1
        self.X = X
        return X

    def betas(self,X,z):
        # Finding coefficients
        beta = np.linalg.pinv(X.T @ X) @ X.T @ z
        return beta

    def split_scale(self,X,z,ts = 0.20):
        self.X_train, self.X_test, self.f_train, self.f_test = train_test_split(X, z, test_size=ts)

    def OLS(self):
        B = self.betas(self.X_train, self.f_train)
        # Setting up model
        f_train_pred = self.X_train @ B
        f_test_pred = self.X_test @ B
        return f_test_pred ,f_train_pred

    def MSE(self, y, y_tilde):
        self.MSE = 1.0/self.n * np.sum((y - y_tilde)**2)
        return self.MSE

    def R2(self, y, y_tilde):
        y_mean = 1.0/self.n * np.sum(y)
        top = np.sum( (y - y_tilde)**2 )
        bottom = np.sum( (y - y_mean)**2 )
        self.R2 = 1 - top/bottom
        return self.R2

    def beta_variance(self, B):
        self.sigma_B = np.var(B)
        return self.sigma_B

    def bootstrap(self,trials):
        z_pred = np.zeros((self.f_train.shape[0], trials))
        for i in range(trials):
            self.X_train, self.f_train = resample(self.X_train, self.f_train)
            z_pred[:,i] = self.OLS()[1]
        return z_pred

    def bias(self, z, z_tilde):
        return np.mean( (z - np.mean(z_tilde, axis = 1))**2 )

    def variance(self, z_tilde):
        return mp.mean( np.var(z_tilde) )

    def bias_variance(self):
        max_complexity = 12
        trials = 100
        complexity = np.linspace(1,max_complexity,max_complexity)
        test_err = np.zeros(len(complexity))
        train_err = np.zeros(len(complexity))

        for i in range(max_complexity):
            test_err[i] = 0
            train_err[i] = 0

            X = self.design_matrix( int(complexity[i]) )
            self.split_scale(X,self.f)
            f_test_pred, f_train_pred = self.OLS()

            test_err[i] += mean_squared_error(self.f_test,f_test_pred)
            train_err[i] += mean_squared_error(self.f_train,f_train_pred)


        plt.plot(complexity, np.log10(train_err), label='Training Error')
        plt.plot(complexity, np.log10(test_err), label='Test Error')
        plt.xlabel('Polynomial degree')
        plt.ylabel('log10[MSE]')
        plt.legend()
        plt.show()


reg = Regression(400)
reg.dataset2D()#mse_train[i] = self.mean_squared_error(y_model[:75],self.f_train)
            #mse_test[i] = self.mean_squared_error(y_model[:25],self.f_test)
X = reg.design_matrix(2)
reg.split_scale(X,reg.f)
B = reg.betas(reg.X_train, reg.f_train)
sigma_B = reg.beta_variance(B)
f_pred, f_tilde = reg.OLS()
print("R2: ", reg.R2(reg.f_train, f_tilde))
print("MSE: ", reg.MSE(reg.f_test, f_pred))
print("Beta variance: ", sigma_B)

reg.bias_variance()

deg=2; boot=100
X = reg.design_matrix(deg)
reg.split_scale(X,reg.f)
f_pred, f_tilde = reg.OLS()
f_pred_boot = reg.bootstrap(boot)
