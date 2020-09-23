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

class Regression():
    def __init__(self, n):
        self.n = n

    def franke_function(self,x,y):
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

        return term1 + term2 + term3 + term4

    def dataset2D(self):
        self.x = np.random.rand(self.n)
        self.y = np.random.rand(self.n)
        noise = 0.1*np.random.randn(self.n)
        self.f = self.franke_function(self.x, self.y) + noise

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

    def OLS(self, X_train, X_test, f_train):
        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ f_train

        f_tilde = X_train @ beta
        f_pred = X_test @ beta

        return f_tilde, f_pred

    def bootstrap(self, X_test, X_train, f_train, trials):
        mse = np.zeros(trials)

        z_pred = np.zeros((self.f_test.shape[0], trials))
        for i in range(trials):
            X_new, f_new = resample(X_train, f_train)
            z_pred[:,i] = self.OLS(X_new, X_test, f_new)[1]
            mse[i] = self.MSE( self.f_test, z_pred[:,i] )
        return z_pred, mse

    def ridge(self, X_train, X_test, f_train, lam):
        beta = np.linalg.pinv(X_train.T @ X_train + np.identity(len(X_train[0,:]))*lam) @ X_train.T @ f_train
        f_tilde = X_train @ beta
        f_pred = X_test @ beta

        return f_tilde, f_pred
    
    def lasso(self,X_train, X_test, f_train, lam):
        lassoreg = Lasso(alpha = lam).predict(X_train, X_test, f_train)
        f_tilde = lassoreg.fit(X_train)
        f_pred = lassoreg.fit(X_test)
        
        return f_tilde, f_pred
        
        

    def k_fold(self,X,k,deg):
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

            f_tilde, z_pred = self.OLS(X_train,X_test,f_train)


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

        return error , np.mean(test_err_arr), np.mean(train_err_arr)


n = 400; maxdeg = 15
degrees = np.linspace(1,maxdeg,maxdeg)

#array for MSE
mse_kfold = np.zeros(maxdeg)
mse_bootstrap = np.zeros(maxdeg)
mse_lasso = np.zeros(maxdeg)

#creating array for train and test error
train_error = np.zeros(maxdeg)
test_error = np.zeros(maxdeg)

train_error_bootstrap = np.zeros(maxdeg)
test_error_bootstrap = np.zeros(maxdeg)

test_error_kfold = np.zeros(maxdeg)
train_error_kfold = np.zeros(maxdeg)

test_error_lasso = np.zeros(maxdeg)
train_error_lasso = np.zeros(maxdeg)

# Arrays for plotting Bias and Variance
bias = np.zeros(maxdeg)
variance = np.zeros(maxdeg)

strap_error = np.zeros(maxdeg)

deg = 2
reg = Regression(n)
reg.dataset2D()
print("max f: ", np.max(reg.f), "min f: ", np.min(reg.f))
reg.design_matrix(deg)
reg.split(reg.X, reg.f)
f_tilde, f_pred = reg.OLS(reg.X_train, reg.X_test, reg.f_train)
print(np.mean( (reg.f_test - f_pred)**2 ))

for i in range(maxdeg):
    deg = int(degrees[i])
    reg = Regression(n)
    reg.dataset2D()
    reg.design_matrix(deg)
    reg.split(reg.X, reg.f)
    f_tilde, f_pred = reg.OLS(reg.X_train, reg.X_test, reg.f_train)

    lam= 0.3
    # Train and Test Error
    test_error[i] = np.mean( (f_pred - reg.f_test)**2 )
    train_error[i] = np.mean( (f_tilde - reg.f_train)**2 )

    # Bootstrap Method for Bias and Variance
    f_strap, mse = reg.bootstrap(reg.X_test, reg.X_train, reg.f_train, trials = 100)
    f_hat = np.mean(f_strap, axis=1, keepdims=True) # Finding the mean for every coloumn element

    strap_error[i] = np.mean( np.mean((reg.f_test.reshape(-1,1) - f_strap)**2, axis=1, keepdims=True) )
    bias[i] = np.mean( (reg.f_test - f_hat)**2 , keepdims=True)
    variance[i] = np.mean(np.var(f_strap, axis=1, keepdims=True))

    #reg.k_fold(reg.X,5,deg)

    #--kfold--
    mse_kfold[i], test_error_kfold[i], train_error_kfold[i] = reg.k_fold(reg.X,5,deg)

    #--lasso--
    f_pred_lasso, f_tilde_lasso  = reg.lasso(reg.X_train, reg.X_test, reg.f_train, lam)

    test_error_lasso[i] = np.mean( (f_pred_lasso - reg.f_test)**2 )
    train_error_lasso[i] = np.mean( (f_tilde_lasso - reg.f_train)**2 )




plt.title('OLS Error')
plt.plot(degrees, train_error, label='Train Error')
plt.plot(degrees, test_error, label='Test Error')
plt.legend()
plt.show()

plt.title('K-fold Error')
plt.plot(degrees, test_error_kfold, label='Test Error')
plt.plot(degrees, train_error_kfold, label='Train Error')
plt.legend()
plt.show()

#plt.title('Lasso Error')
#plt.plot(degrees, test_error_lasso, label='Test Error')
#plt.plot(degrees, train_error_lasso, label='Train Error')
plt.plot(degrees, bias, label='Bias')
plt.plot(degrees, variance, label='Variance')
plt.plot(degrees, strap_error, label='Bootstrap error')
plt.legend()
plt.show()
