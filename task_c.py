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

n = 400; maxdeg = 10
trials = 100 # Bootstraps
lam_value = 0 # Cause we're working with OLS

# Initializing Regression class and setting up dataset with length n
reg = Regression()
reg.dataset_franke(n)

# Creating arrays for plotting
degrees = np.linspace(1,maxdeg,maxdeg, dtype=int) # Complexity array

MSE_test_bootstrap  = np.zeros(maxdeg)

MSE_test_CV  = np.zeros(maxdeg)

# Looping through complexities
for i in range(maxdeg):
    # Setting up design matrix
    deg = degrees[i]
    reg.design_matrix(deg)
    reg.split(reg.X, reg.f)

    # Bootstrap
    f_strap, mse = reg.bootstrap(reg.X_train, reg.X_test, reg.f_train, trials, reg.OLS, lam_value)
    MSE_test_bootstrap[i] = np.mean( np.mean((reg.f_test.reshape(-1,1) - f_strap)**2, axis=1) )

    # Cross-Validation
    MSE_test_CV[i] = reg.k_fold(reg.X,5,deg, reg.OLS)

plt.plot(degrees, MSE_test_bootstrap, label = 'Bootstrap MSE')
plt.plot(degrees, MSE_test_CV, label = 'CV MSE')
plt.legend()
plt.show()
