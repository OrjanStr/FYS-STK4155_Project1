#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from linear_regression import Regression
import seaborn as sns
from linear_regression import coef_plot

"""
plotting heatmap for bootstrap and kfold, and bias variance with ridge
"""

def task_d(maxdeg, lam_lst, x = None, y = None, z = None, data = False):

    n = 400
    degrees = np.linspace(1,maxdeg,maxdeg, dtype = int)
    plot_label_x = [ '%i' % elem for elem in np.log10(lam_lst) ]
    plot_label_y = [ '%i' % elem for elem in degrees ]

    # Arrays for plotting MSE, Bias and Variance
    bias        = np.zeros(len(lam_lst))
    variance    = np.zeros(len(lam_lst))
    strap_error = np.zeros(len(lam_lst))
    # Arrays for plotting MSE as function of lambda and complexity
    deg_lam_error_kfold     = np.zeros((maxdeg,len(lam_lst)))
    deg_lam_error_bootstrap = np.zeros((maxdeg,len(lam_lst)))


    reg = Regression()
    # Setting up our own data via Franke's function if there is no data input
    if data:
        reg.data_setup(x,y,z)
    else:
        reg.dataset_franke(n)

    # Looping through all lambda and complexity values
    for k, lam_value in enumerate(lam_lst):
        for i, deg in enumerate(degrees):
            # Set up design matrix and split into train and test data
            reg.design_matrix(deg)
            reg.split(reg.X, reg.f) # Values are scaled in this method

            # Bootstrap Method for Bias and Variance
            f_strap, mse = reg.bootstrap(reg.X_train, reg.X_test, reg.f_train, trials = 100, method = reg.ridge ,lam = lam_value)
            # Assigning the bootstrap MSE to our plot array
            deg_lam_error_bootstrap[i,k] = np.mean( np.mean((reg.f_test.reshape(-1,1) - f_strap)**2, axis=1) )
            # Calculate k-fold MSE and assign for plot array
            deg_lam_error_kfold[i,k]= reg.k_fold(reg.X,10,reg.ridge,lam_value)

    deg = 10 # Finding bias and variance dependency on lambda for degree 10
    for k, lam_value in enumerate(lam_lst):

        # Bootstrap resampling
        f_strap, mse = reg.bootstrap(reg.X_train, reg.X_test, reg.f_train, trials = 100, method = reg.ridge ,lam = lam_value)
        f_hat = np.mean(f_strap, axis=1) # Finding the mean for every coloumn element

        # Assigning MSE, bias and variance to plot arrays
        strap_error[k] = np.mean( np.mean((reg.f_test.reshape(-1,1) - f_strap)**2, axis=1) )
        bias[k] = np.mean( (reg.f_test - f_hat)**2)
        variance[k] = np.mean(np.var(f_strap, axis=1))

    # Plotting MSE for lambda and complexity
    boot_title = "Ridge MSE: Bootstrap"
    kfold_title = "Ridge MSE: K-fold"
    reg.heatmap(deg_lam_error_bootstrap, boot_title, ticks_x = plot_label_x, ticks_y = plot_label_y,
    save = True , filename = 'boot_heatmap_ridge')
    reg.heatmap(deg_lam_error_kfold, kfold_title, ticks_x = plot_label_x, ticks_y = plot_label_y,
    save = True , filename = 'kfold_heatmap_ridge')

    # Plotting bias and variance for lambda
    reg.single_plot([np.log10(lam_lst), np.log10(lam_lst)], [bias, variance], r'$\lambda$', 'Error',
                 ['Bias','Variance'], 'lambda_bias_variance (deg: %d)' %(deg), save = True, filename = 'lambda_bias_variance_deg%d' %(deg))

if __name__ == "__main__":
    lam_lst = np.logspace(-15,0,20)
    maxdeg = 12
    task_d(maxdeg, lam_lst)

    lam_lst = np.logspace(-4,-1.5,20)
    coef_plot(3,400,lam_lst)
