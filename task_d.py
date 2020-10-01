#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from frankfunc import Regression, heatmap, single_plot
import seaborn as sns


"""
plotting heatmap for bootstrap and kfold, and vias variance with ridge
"""

def task_d(x = None, y = None, z = None, data = False):
    n = 400; maxdeg = 10
    degrees = np.linspace(1,maxdeg,maxdeg, dtype = int)
    lam_lst = np.logspace(-15,-7,20)

    # Arrays for plotting Bias and Variance
    bias = np.zeros(len(lam_lst))
    variance = np.zeros(len(lam_lst))
    strap_error = np.zeros(len(lam_lst))

    deg_lam_error_kfold = np.zeros((maxdeg,len(lam_lst)))
    deg_lam_error_bootstrap = np.zeros((maxdeg,len(lam_lst)))

    reg = Regression()
    if data:
        reg.data_setup(x,y,z)
    else:
        reg.dataset_franke(n)

    for k, lam_value in enumerate(lam_lst):
        for i, deg in enumerate(degrees):
            reg.design_matrix(deg)
            reg.split(reg.X, reg.f)

            # Bootstrap Method for Bias and Variance
            f_strap, mse = reg.bootstrap(reg.X_train, reg.X_test, reg.f_train, trials = 100, method = reg.ridge ,lam = lam_value)


            deg_lam_error_bootstrap[i,k] = np.mean( np.mean((reg.f_test.reshape(-1,1) - f_strap)**2, axis=1) )
            deg_lam_error_kfold[i,k]= reg.k_fold(reg.X,5,reg.ridge,lam_value)
    #lam_lst = np.logspace(1,8,20)
    deg = 8
    for k, lam_value in enumerate(lam_lst):
        f_strap, mse = reg.bootstrap(reg.X_train, reg.X_test, reg.f_train, trials = 100, method = reg.ridge ,lam = lam_value)
        f_hat = np.mean(f_strap, axis=1) # Finding the mean for every coloumn element

        strap_error[k] = np.mean( np.mean((reg.f_test.reshape(-1,1) - f_strap)**2, axis=1) )
        bias[k] = np.mean( (reg.f_test - f_hat)**2)
        variance[k] = np.mean(np.var(f_strap, axis=1))

    sns.heatmap(deg_lam_error_bootstrap)
    plt.title("Bootstrap")
    plt.xlabel("Lambda")
    plt.ylabel("Complexity")
    plt.show()
    print("Plotted!")
    sns.heatmap(deg_lam_error_kfold)
    plt.show()
    #heatmap(lam_lst, degrees, deg_lam_error_bootstrap, "lambda", "complexity", "Bootstrap Error", save = True , filename = 'bootstrap_heatmap')
    #heatmap(lam_lst, degrees, deg_lam_error_kfold,"lambda", "complexity", "K-Fold Error", save = True, filename = 'kfold_heatmap')
    single_plot([lam_lst, lam_lst], [bias, variance], r'$\lambda$', 'Error',
                 ['Bias','Variance'], 'lambda_bias_variance (deg =: %d)' %(deg), save = True, filename = 'lambda_bias_variance_deg%d' %(deg))

if __name__ == "__main__":
    task_d()
