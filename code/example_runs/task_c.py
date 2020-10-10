import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0,"..")
from linear_regression import Regression
from sklearn.model_selection import KFold
"""
compare bootstrap with kfold for OLS
"""

def task_c(maxdeg, x=None, y=None, z=None, data=False):
    n = 400;
    trials = 100 # Bootstraps
    lam_value = 0 # Cause we're working with OLS

    # Initializing Regression class and setting up dataset if there's no input
    reg = Regression()
    if data:
        reg.data_setup(x,y,z)
    else:
        reg.dataset_franke(n)

    # Creating arrays for plotting
    degrees = np.linspace(1,maxdeg,maxdeg, dtype=int) # Complexity array
    MSE_test_bootstrap  = np.zeros(maxdeg)
    MSE_test_CV  = np.zeros(maxdeg)
    MSE_test_sklearn = np.zeros(maxdeg)

    # Looping through complexities
    for i in range(maxdeg):
        deg = degrees[i]
        # Setting up design matrix
        reg.design_matrix(deg)
        reg.split(reg.X, reg.f)

        # Bootstrap
        f_strap = reg.bootstrap(reg.X_train, reg.X_test, reg.f_train, trials, reg.OLS, lam_value)
        MSE_test_bootstrap[i] = np.mean( np.mean((reg.f_test.reshape(-1,1) - f_strap)**2, axis=1) )

        # Cross-Validation
        MSE_test_CV[i] = reg.k_fold(reg.X,5, reg.OLS, lam=0)

        kfold = KFold(n_splits = 5)
        for train, test in kfold.split(reg.X):
            Xtrain, Xtest = reg.X[train], reg.X[test]
            ftrain, ftest =  reg.f[train], reg.f[test]
            f_tilde, f_pred = reg.OLS(Xtrain, Xtest, ftrain)

            MSE_test_sklearn[i] = np.mean( (f_pred - ftest)**2 )

    # Plotting the two MSEs
    reg.single_plot((degrees, degrees), (MSE_test_bootstrap, MSE_test_CV), 'Complexity', 'MSE', ('Bootstrap', 'K-fold'),
    'Bootstrap MSE vs. K-fold MSE for OLS', save = True, filename = 'Kfold_error_OLS')

if __name__ == "__main__":
    task_c(maxdeg = 10)
