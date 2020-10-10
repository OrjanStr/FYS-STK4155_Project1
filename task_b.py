import numpy as np
from linear_regression import Regression

"""
using OLS and bootstrap to produce Training error, Test error plot
and Bias Variance plot.
"""


def task_b(maxdeg, x = None , y = None , z = None, data = False):
    n = 400
    degrees = np.linspace(1,maxdeg,maxdeg, dtype = int)

    # Arrays for plotting error
    train_error = np.zeros(maxdeg)
    test_error = np.zeros(maxdeg)

    # Arrays for plotting Bias and Variance
    bias = np.zeros(maxdeg)
    variance = np.zeros(maxdeg)
    strap_error = np.zeros(maxdeg)
    reg = Regression()

    # Generating own data from Franke's function if there's no input
    if data:
        reg.data_setup(x,y,z)
    else:
        reg.dataset_franke(n)

    # Looping through complexities
    for i, deg in enumerate(degrees):
        np.random.seed(42) #For same split every time

        reg.design_matrix(deg)
        reg.split(reg.X, reg.f)

        f_tilde, f_pred = reg.OLS(reg.X_train, reg.X_test, reg.f_train)
        # Adding train and test error to plotting arrays
        test_error[i] = np.mean( (f_pred - reg.f_test)**2 )
        train_error[i] = np.mean( (f_tilde - reg.f_train)**2 )

        # Bootstrap Method
        f_strap = reg.bootstrap(reg.X_train, reg.X_test, reg.f_train,
                                     trials = 100, method = reg.OLS ,lam = 0)
        f_hat = np.mean(f_strap, axis=1) # Finding the mean for every coloumn element

        # Calculating MSE, bias and variance
        strap_error[i] = np.mean( np.mean((reg.f_test.reshape(-1,1) - f_strap)**2, axis=1) )
        bias[i] = np.mean( (reg.f_test - f_hat)**2)
        variance[i] = np.mean(np.var(f_strap, axis=1))


    reg.single_plot([degrees, degrees], [train_error, test_error],
                'Complexity', 'Error', ['Train Error', 'Test Error'],
                'OLS Error', save = True, filename = 'OLS_error')

    reg.single_plot([degrees, degrees, degrees], [bias, variance, strap_error],
                'complexity', 'Error', ['Bias', 'Variance', 'Bootstrap Error'],
                'Bias, Variance, Error', save = True, filename = 'bias_variance')

if __name__ == "__main__":
    task_b(maxdeg = 10)
