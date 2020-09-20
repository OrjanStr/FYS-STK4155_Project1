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

class Regression():
    def __init__(self, n):
        self.n = n

    def franke_function(self,x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

        return term1 + term2 + term3 + term4

    def dataset2D(self):
        self.x = np.random.rand(self.n)
        self.y = np.random.rand(self.n)
        noise = 0.1*np.random.randn(self.n)
        self.f = self.franke_function(self.x, self.y) + noise

    def design_matrix(self, deg):
        # features
        p = int(0.5*( (deg+1)*(deg+2) ))
        X = np.zeros((self.n,p))
        idx=0
        for i in range(deg+1):
            for j in range(deg+1-i):
                X[:,idx] = self.x**i * self.y**j
                idx += 1
        self.X = X
        return X

    def split(self, X, f, scale=True):
        # Scaling Data
        if scale:
            X[:,1:] -= np.mean(X[:,1:], axis=0)
            f -= np.mean(f)
        # Splitting Data
        self.X_train, self.X_test, self.f_train, self.f_test = train_test_split(X,f,test_size=0.2)
        return self.X_train, self.X_test, self.f_train, self.f_test

    def OLS(self, X_train, X_test, f_train):
        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ f_train

        f_tilde = X_train @ beta
        f_pred = X_test @ beta

        return f_tilde, f_pred

    def bootstrap(self, X_test, X_train, f_train, trials):
        z_pred = np.zeros((self.f_test.shape[0], trials))
        for i in range(trials):
            X_train, f_train = resample(X_train, f_train)
            z_pred[:,i] = self.OLS(X_train, X_test, f_train)[1]
        return z_pred

n = 50000; maxdeg = 10
# Arrays for plotting error
degrees = np.linspace(1,maxdeg,maxdeg)
train_error = np.zeros(maxdeg)
test_error = np.zeros(maxdeg)
# Arrays for plotting Bias and Variance
bias = np.zeros(maxdeg)
variance = np.zeros(maxdeg)
bias2 = np.zeros(maxdeg)
variance2 = np.zeros(maxdeg)

deg = 2
reg = Regression(n)
reg.dataset2D()
print("max x: ", np.max(reg.f), "min x: ", np.min(reg.f))
reg.design_matrix(deg)
reg.split(reg.X, reg.f)
f_tilde, f_pred = reg.OLS(reg.X_train, reg.X_test, reg.f_train)
print(np.mean( (reg.f_test - f_pred)**2 ))

for i in range(maxdeg):
    deg = int(degrees[i])
    reg = Regression(n)
    reg.dataset2D()
    reg.design_matrix(deg)
    reg.split(reg.X, reg.f)
    f_tilde, f_pred = reg.OLS(reg.X_train, reg.X_test, reg.f_train)

    # Train and Test Error
    test_error[i] = np.mean( (f_pred - reg.f_test)**2 )
    train_error[i] = np.mean( (f_tilde - reg.f_train)**2 )

    # Bootstrap Method for Bias and Variance
    f_strap = reg.bootstrap(reg.X_test, reg.X_train, reg.f_train, trials = 100)
    f_hat = np.mean(f_strap, axis=1, keepdims=True) # Finding the mean for every coloumn element

    bias[i] = np.mean( (reg.f_test - np.mean(f_hat))**2 )
    variance[i] = np.mean(np.var(f_strap, axis=0))

    bias2[i] = np.mean( (reg.f_test - np.mean(f_hat))**2 )
    variance2[i] = np.mean(np.var(f_strap, axis=1, keepdims=True))

plt.plot(degrees, train_error, label='Training Error')
plt.plot(degrees, test_error, label='Test Error')
plt.legend()
plt.show()

plt.plot(degrees, bias, label='Bias')
plt.plot(degrees, variance, label='Variance')
plt.legend()
plt.show()
