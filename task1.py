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

def task_a(x_set=None,y_set=None,z_set=None, data=False):
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
    if data:
        reg.data_setup(x_set,y_set,z_set)
    else:
        reg.dataset_franke(n)

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
        if data:
            o2 = 1 # Assume variance of noise is 1 when noise is unknown
        else:
            o2 = reg.o2

        if deg == 5: # Pick out the beta variance in complexity 5
            p = reg.X_train.shape[1] # Number of features
            print(p)
            betas = reg.beta_OLS
            beta_variance = reg.beta_confidence(reg.beta_OLS, o2, reg.X_train,n)

    # Plotting betas with confidence intervals for complexity 5
    features = np.linspace(0,p-1,p, dtype=int)
    plt.style.use('seaborn-whitegrid')
    plt.errorbar(features, betas,  yerr=beta_variance, fmt=".k", capsize=3)
    plt.xticks(features, my_xticks, fontsize='12')
    plt.yticks(fontsize='12')
    plt.title("Beta Variance OLS | Complexity  = 5", fontsize = '16')
    plt.xlabel("Features",fontsize = '16')
    plt.ylabel(r"$\beta$",fontsize = '16')
    plt.show()

    # Plotting MSE and R2 score as function of complexity
    plt.plot(degrees, MSE, label = 'MSE', linewidth=3)
    plt.plot(degrees, R2, label = 'R2', linewidth=3)
    plt.title("MSE and R2 score for OLS", fontsize = '16')
    plt.tick_params(labelsize='12')
    plt.xlabel("Complexity", fontsize = '16')
    plt.ylabel("Score", fontsize = '16')
    plt.legend(fontsize='12', loc=7)
    plt.show()

if __name__ == "__main__":
    task_a()
