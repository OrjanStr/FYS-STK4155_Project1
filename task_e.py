from random import seed
import numpy as np
from linear_regression import Regression, Visuals


def task_e(maxdeg,lam_lst, trials, x = None, y = None, z = None, data = False):
    n=400; maxdeg=maxdeg; trials=trials
    lam_lst = np.logspace(-15,-8,20)
    degrees = np.linspace(1,maxdeg,maxdeg, dtype=int)
    
    # Arrays for plotting MSE, Bias and Variance (bootstrap)
    bias = np.zeros((maxdeg, len(lam_lst)))
    variance = np.zeros((maxdeg, len(lam_lst)))
    strap_error = np.zeros((maxdeg, len(lam_lst)))
    
    # Array for plotting MSE for Cross-Validation
    CV_error = np.zeros((maxdeg,len(lam_lst)))
    
    reg = Regression()
    if data:
        reg.data_setup(x,y,z)
    else:
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
    
    vs = Visuals()
    vs.heatmap(CV_error, 'Lasso MSE - Cross-Validation', save = True, filename = 'LassoMSE_CV')
    vs.heatmap(strap_error, 'Lasso MSE - Bootstrap', save = True, filename = 'LassoMSE_Bootstrap')
    vs.heatmap(bias, 'Lasso Bias', save = True, filename = 'Lasso_bias')
    vs.heatmap(variance, 'Lasso Variance', save = True, filename = 'Lasso_variance')

if __name__ == "__main__":
    task_e(10,np.logspace(-15,-8,20),100)