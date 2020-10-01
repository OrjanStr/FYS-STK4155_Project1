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
from sklearn.linear_model import Lasso
from sklearn import linear_model
from frankfunc import Regression
from frankfunc import heatmap


n=400; maxdeg=10; trials=100;
lam_lst = np.logspace(-15,-8,20)
degrees = np.linspace(1,maxdeg,maxdeg, dtype=int)

# Arrays for plotting MSE, Bias and Variance (bootstrap)
bias = np.zeros((maxdeg, len(lam_lst)))
variance = np.zeros((maxdeg, len(lam_lst)))
strap_error = np.zeros((maxdeg, len(lam_lst)))

# Array for plotting MSE for Cross-Validation
CV_error = np.zeros((maxdeg,len(lam_lst)))

reg = Regression()
reg.dataset_franke(n)

for k, lam_value in enumerate(lam_lst):
    for i in range(maxdeg):
        deg = degrees[i]
        reg.design_matrix(deg)
        reg.split(reg.X, reg.f, scale=False)

        # Bootstrap for MSE, Bias and Variance
        f_strap, mse = reg.bootstrap(reg.X_train, reg.X_test, reg.f_train, trials, reg.lasso, lam_value)
        f_hat = np.mean(f_strap, axis=1) # Finding the mean for every coloumn element

        strap_error[i,k] = np.mean( np.mean((reg.f_test.reshape(-1,1) - f_strap)**2, axis=1) )
        bias[i,k] = np.mean( (reg.f_test - f_hat)**2)
        variance[i,k] = np.mean(np.var(f_strap, axis=1))

        # Cross validation for MSE
        CV_error[i,k] = reg.k_fold(reg.X, 5, reg.lasso, lam_value)


heatmap(lam_lst, degrees, CV_error, r'$\lambda$', 'Complexity', 'Lasso MSE - Cross-Validation')
heatmap(lam_lst, degrees, strap_error, r'$\lambda$', 'Complexity', 'Lasso MSE - Bootstrap')
heatmap(lam_lst, degrees, bias, r'$\lambda$', 'Complexity', 'Lasso Bias')
heatmap(lam_lst, degrees, variance, r'$\lambda$', 'Complexity', 'Lasso Variance')
