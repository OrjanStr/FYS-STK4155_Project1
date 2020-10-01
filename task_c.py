import matplotlib.pyplot as plt
import numpy as np
from frankfunc import Regression
from frankfunc import single_plot

def task_c(x=None, y=None, z=None, data=False):
    n = 400; maxdeg = 10
    trials = 100 # Bootstraps
    lam_value = 0 # Cause we're working with OLS

    # Initializing Regression class and setting up dataset with length n
    reg = Regression()
    if data:
        reg.data_setup(x,y,z)
    else:
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
        MSE_test_CV[i] = reg.k_fold(reg.X,5, reg.OLS, lam=0)

    plt.plot(degrees, MSE_test_bootstrap, label = 'Bootstrap MSE')
    #plt.plot(degrees, MSE_test_CV, label = 'CV MSE')
    plt.legend()
    plt.show()

    n = 400; deg = 5
    folds = [5,6,7,8,9,10]

    # Arrays for plotting Bias and Variance
    bias = np.zeros(len(folds))
    variance = np.zeros(len(folds))
    error = np.zeros(len(folds))


    reg = Regression()
    if data:
        reg.data_setup(x,y,z)
    else:
        reg.dataset_franke(n)
    reg.design_matrix(deg)
    reg.split(reg.X, reg.f)
    for i, fold in enumerate(folds):
        error[i]= reg.k_fold(reg.X,fold,reg.OLS,0)
    single_plot([folds], [error], 'folds', 'error', ' ', 'title')

if __name__ == "__main__":
    task_c()
