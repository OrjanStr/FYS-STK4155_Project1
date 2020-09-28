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

n = 400; deg = 3
reg = Regression(n)
reg.dataset2D()
print("max f: ", np.max(reg.f), "min f: ", np.min(reg.f))
reg.design_matrix(deg)
reg.split(reg.X, reg.f)
f_tilde, f_pred = reg.OLS(reg.X_train, reg.X_test, reg.f_train)
print(np.mean( (reg.f_test - f_pred)**2 ))

degrees = np.linspace(1,5,5, dtype=int)

def R_squared(y,y_tilde):
    top = np.mean( (y - y_tilde)**2 )
    bot = np.mean( (y - np.mean(y))**2 )
    return 1 - top/bot

R2 = np.zeros(5)
MSE = np.zeros(5)

reg = Regression(n)
reg.dataset2D()
for i in range(5):
    deg = degrees[i]
    reg.design_matrix(deg)
    reg.split(reg.X, reg.f)
    f_tilde, f_pred = reg.OLS(reg.X_train, reg.X_test, reg.f_train)
    MSE[i] = np.mean( (reg.f_test - f_pred)**2 )
    R2[i] = R_squared(reg.f_test, f_pred)

plt.plot(degrees, MSE, label = 'MSE')
plt.plot(degrees, R2, label = 'R2')
plt.legend()
plt.show()
