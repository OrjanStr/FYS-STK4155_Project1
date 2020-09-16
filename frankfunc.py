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



class Regression:
    """
        Parameters
        ----------
        n : int
            DESCRIPTION.





        Examples
        --------
    """

    def __init__(self,n):

        self.n = n

    def _franke_function(self,x,y):
        """
        Parameters
        ----------
        x : array
            DESCRIPTION.
        y : array
            DESCRIPTION.
        Returns
        -------
        array
            DESCRIPTION.
        """

        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

        return term1 + term2 + term3 + term4

    def plot_franke(self):
        """

        Returns
        -------
        None.
        """
        # Make data.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x = np.arange(0, 1, 0.05)
        y = np.arange(0, 1, 0.05)
        x, y = np.meshgrid(x,y)
        z = self._franke_function(x, y)

        # Plot the surface.
        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
        # Customize the z axis.
        ax.set_zlim(-0.10, 1.40)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

    def dataset2D(self):
        """

        Returns
        -------
        None.
        """
        # Setting up dataset
        self.x = np.random.randn(self.n)
        self.y = np.random.randn(self.n)

        # Setting up the FrankeFunction with added noise
        noise = 0.1*np.random.randn(self.n)

        self.f = self._franke_function(self.x, self.y) + noise

        # Calculating variance of noise for later use
        self.sigma_squared = 1.0/self.n * np.sum( noise**2 )

    def design_matrix(self):
        # Setting up design matrix
        poly = PolynomialFeatures(self.deg)
        X = np.zeros((self.n,2))
        X[:,0] = self.x[0]
        X[:,1] = self.y[:,0]
        self.X = poly.fit_transform(X)

    def linear_regression(self, ts=0.25):

        # Splitting into train and test data
        self.X_train, self.X_test, self.f_train, self.f_test = train_test_split(self.X, self.f, test_size=ts)

        # Linear Regression

        linreg = LinearRegression()
        linreg.fit(self.X_train, self.f_train)
        self.f_predict = linreg.predict(self.X)

        self.y_train_pred = linreg.predict(self.X_train)
        self.y_test_pred = linreg.predict(self.X_test)
        return self.f_predict

    def design_matrix_homemade(self, deg):
        """

        Parameters
        ----------
        deg : int
            DESCRIPTION.
        Returns
        -------
        X numpy.ndarray
            DESCRIPTION.
        """
        # Setting up matrix
        self.p = int(0.5 * (deg + 1) * (deg + 2))
        X = np.zeros((self.n, self.p))
        # Filling in values
        idx = 0
        for i in range(deg+1):
            for j in range(deg+1-i):
                X[:,idx] = self.x**i * self.y**j
                idx += 1


        return X

    def betas(self,X,z):
        """

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        z : TYPE
            DESCRIPTION.
        Returns
        -------
        beta : TYPE
            DESCRIPTION.
        """
        # Finding coefficients
        beta = np.linalg.pinv(X.T @ X) @ X.T @ z
        return beta

    def split_data(self,X,z,ts = 0.20):
        """

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        z : TYPE
            DESCRIPTION.
        ts : float, optional
            DESCRIPTION. The default is 0.20.
        Returns
        -------
        None.
        """
        self.X_train, self.X_test, self.f_train, self.f_test = train_test_split(X, z, test_size=ts)


    def linear_regression_homemade(self):
        """

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        Returns
        -------
        f_test_pred : TYPE
            DESCRIPTION.
        f_train_pred : TYPE
            DESCRIPTION.
        """
        B = self.betas(self.X_train, self.f_train)
        # Setting up model
        f_train_pred = self.X_train @ B
        f_test_pred = self.X_test @ B
        return f_test_pred ,f_train_pred

    def mean_squared_error_homemade(self, y, y_tilde):
        """

        Parameters
        ----------
        y : TYPE
            DESCRIPTION.
        y_tilde : TYPE
            DESCRIPTION.
        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        self.MSE = 1.0/self.n * np.sum((y - y_tilde)**2)
        return self.MSE

    def r_squared(self, y, y_tilde):
        """

        Parameters
        ----------
        y : TYPE
            DESCRIPTION.
        y_tilde : TYPE
            DESCRIPTION.
        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        y_mean = 1.0/self.n * np.sum(y)
        top = np.sum( (y - y_tilde)**2 )
        bottom = np.sum( (y - y_mean)**2 )
        self.R2 = 1 - top/bottom
        return self.R2

    def beta_variance(self, B):
        """

        Parameters
        ----------
        B : TYPE
            DESCRIPTION.
        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        self.sigma_B = np.var(B)
        return self.sigma_B

    def bootstrap(self,trials):
        z_pred = np.zeros((self.f_train.shape[0], trials))
        for i in range(trials):
            self.X_train, self.f_train = resample(self.X_train, self.f_train)
            z_pred[:,i] = self.linear_regression_homemade()[1]
        return z_pred

    def bias(self, z, z_tilde):
        return np.mean( (z - np.mean(z_tilde, axis = 1))**2 )

    def variance(self, z_tilde):
        return mp.mean( np.var(z_tilde) )

    def bias_variance(self):
        """

        Returns
        -------
        None.
        """
        max_complexity = 12
        trials = 100
        complexity = np.linspace(1,max_complexity,max_complexity)
        test_err = np.zeros(len(complexity))
        train_err = np.zeros(len(complexity))

        for deg in range(1,max_complexity):
            test_err[deg] = 0
            train_err[deg] = 0

            X = self.design_matrix_homemade(deg)
            self.split_data(X,self.f)
            f_test_pred, f_train_pred = self.linear_regression_homemade()

            test_err[deg] += mean_squared_error(self.f_test,f_test_pred)
            train_err[deg] += mean_squared_error(self.f_train,f_train_pred)


        plt.plot(complexity, np.log10(train_err), label='Training Error')
        plt.plot(complexity, np.log10(test_err), label='Test Error')
        plt.xlabel('Polynomial degree')
        plt.ylabel('log10[MSE]')
        plt.legend()
        plt.show()


reg = Regression(400)
reg.dataset2D()#mse_train[i] = self.mean_squared_error(y_model[:75],self.f_train)
            #mse_test[i] = self.mean_squared_error(y_model[:25],self.f_test)
X = reg.design_matrix_homemade(2)
reg.split_data(X,reg.f)
B = reg.betas(reg.X_train, reg.f_train)
sigma_B = reg.beta_variance(B)
f_pred, f_tilde = reg.linear_regression_homemade()
print("R2: ", reg.r_squared(reg.f_train, f_tilde))
print("MSE: ", reg.mean_squared_error_homemade(reg.f_test, f_pred))
print("Beta variance: ", sigma_B)

reg.bias_variance()

deg=2; boot=100
X = reg.design_matrix_homemade(deg)
reg.split_data(X,reg.f)
f_pred, f_tilde = reg.linear_regression_homemade()
f_pred_boot = reg.bootstrap(boot)
