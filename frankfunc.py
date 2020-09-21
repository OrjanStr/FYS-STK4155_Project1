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

    def split_scale(self,X,z,ts = 0.20, scale=True):
        if scale:
            X[:,1:] -= np.mean(X[:,1:], axis=0)
            z -= np.mean(z)
        self.X_train, self.X_test, self.f_train, self.f_test = train_test_split(X, z, test_size=ts)

    def OLS(self, X_train, X_test, f_train):
        B = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ f_train
        # Setting up model
        f_train_pred = X_train @ B
        f_test_pred = X_test @ B
        return f_test_pred ,f_train_pred

    def MSE(self, y, y_tilde):
        error = np.mean((y - y_tilde)**2)
        return error

    def R2(self, y, y_tilde):
        y_mean = np.mean(y)
        top = np.sum( (y - y_tilde)**2 )
        bottom = np.sum( (y - y_mean)**2 )
        self.R2 = 1 - top/bottom
        return self.R2

    def beta_variance(self):
        o2 = self.sigma_squared
        X_ = self.X_train
        self.sigma_B = o2 * np.linalg.pinv( X_.T @ X_ )
        print("beta shape: ", self.sigma_B.shape)
        return self.sigma_B

    def bootstrap(self,trials):
        z_pred = np.zeros((self.f_train.shape[0], trials))
        for i in range(trials):
            X_new, f_new = resample(self.X_train, self.f_train)
            z_pred[:,i] = self.OLS(X_new, self.X_test, f_new)[1]
        return z_pred

    def confidence_interval(self, perc, trials):
        z_pred = self.bootstrap(trials) # Bootstrap sampling
        means = np.mean(z_pred, axis=0) # Calculate the mean of each coloumn (each prediction)
        lower = np.percentile(means, 100 - perc)
        upper = np.percentile(means, perc)
        return np.array([lower, upper])

    def bias(self, z, z_tilde):
        return np.mean( (z - np.mean(z_tilde))**2 )

    def variance(self, z_tilde):
        return np.var(z_tilde)

if __name__ == "__main__":
    n=1000; deg=2

    reg = Regression(500)
    reg.dataset2D()
    X = reg.design_matrix(deg)
    reg.split_scale(X,reg.f)
    f_pred, f_tilde = reg.OLS(reg.X_train, reg.X_test, reg.f_train)
    B_var = reg.beta_variance()

    print("R2: ", reg.R2(reg.f_train, f_tilde))
    print("MSE: ", reg.MSE(reg.f_test, f_pred))
    print("95% Confidence Interval: ", reg.confidence_interval(95,100))

    max_complexity = 10
    trials = 100
    boots = 1000
    complexity = np.linspace(1,max_complexity,max_complexity)

    test_err = np.zeros(len(complexity))
    train_err = np.zeros(len(complexity))

    bias = np.zeros(len(complexity))
    variance = np.zeros(len(complexity))

    for i in range(max_complexity):
        reg = Regression(n)
        reg.dataset2D() # Set up data
        X = reg.design_matrix( int(complexity[i]) ) # Create design matrix
        reg.split_scale(X,reg.f) # Split and scale data
        f_test_pred, f_train_pred = reg.OLS(reg.X_train, reg.X_test, reg.f_train)

        test_err[i] = reg.MSE(reg.f_test, f_test_pred)
        train_err[i] = reg.MSE(reg.f_train,f_train_pred)

        z_pred = reg.bootstrap(boots) # Running bootstrap resampling
        f_hat = np.mean(z_pred, axis=1) # Finding optimal parameter by taking mean of samples
        bias[i] = reg.bias(reg.f_test, f_hat)
        variance[i] = np.mean(np.var(z_pred, axis=0))

    plt.plot(complexity, train_err, label='Training Error')
    plt.plot(complexity, test_err, label='Test Error')
    plt.xlabel('Complexity')
    plt.ylabel('MSE')
    plt.legend()
    plt.title("Training vs Test Error (MSE): n = %i"%n)
    plt.show()

    plt.plot(complexity, bias, label='Bias')
    plt.plot(complexity, variance, label='Variance')
    plt.xlabel('Complexity')
    plt.legend()
    plt.title("Bias-Variance Tradeoff: n=%i, bootstraps=%i"%(n,boots))
    plt.show()