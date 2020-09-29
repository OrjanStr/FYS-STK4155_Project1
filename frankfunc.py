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

class Regression():

    def franke_function(self,x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

        return term1 + term2 + term3 + term4

    def dataset_franke(self,n):
        self.x = np.random.rand(n)
        self.y = np.random.rand(n)
        noise = 0.1*np.random.randn(n)
        self.f = self.franke_function(self.x, self.y) + noise

        self.o2 = np.var(noise)
        self.n = n
    def data_setup(self,x,y,z):
        self.n = len(x)
        self.x ,self.y, self.f = x,y,z

    def design_matrix(self, deg):
        # features
        p = int(0.5*( (deg+1)*(deg+2) ))
        X = np.zeros((self.n,p))
        idx=0
        for i in range(deg+1):
            for j in range(deg+1-i):
                X[:,idx] = self.x**i * self.y**j
                idx += 1
        self.X = X
        return X

    def split(self, X, f, scale=True):
        # Scaling Data
        if scale:
            X[:,1:] -= np.mean(X[:,1:], axis=0)
            f -= np.mean(f)
        # Splitting Data
        self.X_train, self.X_test, self.f_train, self.f_test = train_test_split(X,f,test_size=0.2)
        return self.X_train, self.X_test, self.f_train, self.f_test


    def MSE(self,y,y_pred):
        mse = np.mean((y - y_pred)**2)
        return mse

    def OLS(self, X_train, X_test, f_train, lam=0):
        if lam>0:
            raise ValueError('You are trying to use OLS with a lambda value greater than 0')


        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ f_train

        f_tilde = X_train @ beta
        f_pred = X_test @ beta

        return f_tilde, f_pred

    def bootstrap(self, X_train, X_test, f_train, trials, method, lam):
        mse = np.zeros(trials)

        z_pred = np.zeros((self.f_test.shape[0], trials))
        for i in range(trials):
            X_new, f_new = resample(X_train, f_train)
            z_pred[:,i] = method(X_new, X_test, f_new, lam)[1]
            mse[i] = self.MSE( self.f_test, z_pred[:,i] )
        return z_pred, mse

    def beta_confidence(self, beta, o2, X_train,n):
        # Calculating variance
        var_beta_matrix = o2 * np.linalg.pinv(X_train.T @ X_train)
        var_beta = np.diagonal(var_beta_matrix)

        # Calculating confidence interval
        self.confidence = np.zeros((var_beta.shape[0], 2))
        self.confidence[:,0] = beta - 1.96 * 1/np.sqrt(n) * var_beta
        self.confidence[:,1] = beta + 1.96 * 1/np.sqrt(n) * var_beta
        return self.confidence

    def ridge(self, X_train, X_test, f_train, lam):
        beta = np.linalg.pinv(X_train.T @ X_train + np.identity(len(X_train[0,:]))*lam) @ X_train.T @ f_train
        f_tilde = X_train @ beta
        f_pred = X_test @ beta

        self.beta_for_plot_ridge = beta

        return f_tilde, f_pred

    def lasso(self,X_train, X_test, f_train, lam):

        lassoreg = Lasso(alpha = lam)
        lassoreg.fit(X_train,f_train)
        f_tilde = lassoreg.predict(X_train)
        f_pred = lassoreg.predict(X_test)

        self.beta_for_plot_lasso = lassoreg.coef_

        return f_tilde, f_pred





    def k_fold(self,X,k,deg, method, lam):
        # X = np.random.shuffle(X_train)

        #scaling data
        X[:,1:] -= np.mean(X[:,1:], axis=0)
        f = self.f - np.mean(self.f)

        #splitting data
        X_lst1 = np.split(X,k)
        f_lst1 = np.split(f,k)

        X_lst = []
        f_lst = []

        mse_score = np.zeros(k)
        test_err_arr = np.zeros(k)
        train_err_arr = np.zeros(k)

        #getting rid of nested list
        for i in range(len(X_lst1)):
            X_lst.append(X_lst1[i])
            f_lst.append(f_lst1[i])

        X_lst = np.array(X_lst)
        f_lst = np.array(f_lst)

        for i in range(len(X_lst)):

            X_train = np.concatenate([X_lst[:i],X_lst[i+1:]])
            f_train = np.concatenate([f_lst[:i],f_lst[i+1:]])
            X_test = X_lst[i]
            f_test = f_lst[i]


            f_train_lst = []
            X_train_lst = []

            # do this in the other loop?
            for k in range(len(X_train)):
                for j in range(len(X_train[k])):
                    f_train_lst.append(f_train[k][j])
                    X_train_lst.append(X_train[k][j])

            f_train = np.array(f_train_lst)
            X_train = np.array(X_train_lst)

            f_tilde, z_pred = method(X_train,X_test,f_train,lam)


            # mse_score[i] = self.mse(f_test,f_pred)
            mse_score[i] = self.MSE(f_test,z_pred)


            # bias[i] = np.mean((f_test - np.mean(z_pred))**2)
            average_model = np.mean(z_pred)

            test_err_arr[i] = np.mean( (z_pred - f_test)**2 )
            train_err_arr[i] = np.mean( (f_tilde - f_train)**2 )




        error = np.mean(mse_score)

        # print ('test_error', np.mean(test_err_arr))
        # print ('train_error',np.mean(train_err_arr))
        # print ('deg',deg,'score',error,'\n')

        return np.mean(test_err_arr)



def heatmap(x, y, z, label_x, label_y, title, save = False, filename = None):

    fig1, ax1 = plt.subplots()
    cs = ax1.contourf(x, y, z, cmap ='Greens', extend ='both', alpha = 1)
    fig1.colorbar(cs)
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    ax1.set_title(title)
    if save:
        plt.savefig('visuals/' + filename + '.pdf')
    plt.show()

def single_plot(x,y, label_x, label_y, func_label, title, save = False, filename = None):

    plt.title(title)

    for i in range(len(x)):
        plt.plot(x[i], y[i], label = func_label[i])

    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    if save:
        plt.savefig('visuals/' + filename + '.pdf')
    plt.show()


def compare_plot(x, y, label_x, label_y,
                 graph_label, title, subtitle, save = False, filename = None):

    fig, axs = plt.subplots(2)
    fig.suptitle(title)

    axs[0].plot(x[0], y[0])
    axs[0].title.set_text(subtitle[0])
    axs[0].set_xlabel(label_x[0])
    axs[0].set_ylabel(label_y[0])


    axs[1].plot(x[1], y[1])
    axs[1].title.set_text(subtitle[1])
    axs[1].set_xlabel(label_x[1])
    axs[1].set_ylabel(label_y[1])

    if save:
        plt.savefig('visuals/' + filename + '.pdf')
    plt.legend()
    plt.show()



def coef_plot(deg, n, lam_lst):
    reg.dataset_franke(n)
    reg.design_matrix(deg)

    ridge_coefs = []
    lasso_coefs = []

    for lamb in lam_lst:
        reg.ridge(reg.X_train, reg.X_test, reg.f_train, lamb)
        ridge_coefs.append(reg.beta_for_plot_ridge)

        reg.lasso(reg.X_train, reg.X_test, reg.f_train, lamb)
        lasso_coefs.append(reg.beta_for_plot_lasso)

    fig, axs = plt.subplots(2)
    fig.suptitle('Ridge and Lasso as lambda increases')
    axs[0].plot(lam_lst[:18], ridge_coefs[:18])
    axs[1].plot(lam_lst[:10], lasso_coefs[:10])
    plt.show()




if __name__ == "__main__":
    1+1
