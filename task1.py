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

def task_a(x_set=None,y_set=None,z_set=None, generate=False):
    n = 400; deg = 3

    def R_squared(y,y_tilde):
        top = np.mean( (y - y_tilde)**2 )
        bot = np.mean( (y - np.mean(y))**2 )
        return 1 - top/bot

    # Arrays for R2 and MSE for complexities 1-5
    degrees = np.linspace(1,5,5, dtype=int)
    R2 = np.zeros(5)
    MSE = np.zeros(5)

    # Initializing Regression class
    reg = Regression()
    # Checking if we need to generate our own dataset
    if generate:
        reg.dataset_franke(n)
    else:
        reg.data_setup(x_set,y_set,z_set)

    # Looping through complexities
    for i in range(5):
        # Make prediction
        deg = degrees[i]
        reg.design_matrix(deg)
        reg.split(reg.X, reg.f)
        f_tilde, f_pred = reg.OLS(reg.X_train, reg.X_test, reg.f_train)

        # Calculate errors
        MSE[i] = np.mean( (reg.f_test - f_pred)**2 )
        R2[i] = R_squared(reg.f_test, f_pred)

        # Confidence interval
        if generate:
            o2 = reg.o2
        else:
            o2 = 1 # Assume variance of noise is 1 when nosie is unknown

        confidence = reg.beta_confidence(reg.beta_OLS, o2, reg.X_train,n)

    plt.plot(degrees, MSE, label = 'MSE')
    plt.plot(degrees, R2, label = 'R2')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    task_a()
