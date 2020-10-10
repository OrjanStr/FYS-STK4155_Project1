import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import Lasso
import seaborn as sb
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class Regression():
    """
    class to handle Linear regression with a dataset.
    can also generate from franke's function.
    """

    def __init__(self):
        #defined in dataset
        self.x = None
        self.y = None
        self.f = None
        self.n = None
        self.o2 = None
        self.terrain = False

        #defined in design matrix
        self.X = None

        #defined in split
        self.X_train = None
        self.X_test = None
        self.f_train = None
        self.f_test = None

        #defined in beta_confidence
        self.confidence = None

        #defined in lasso
        self.beta_for_plot_lasso = None

        #defined in ridge
        self.beta_for_plot_ridge = None

        #defined in OLS
        self.beta_OLS = None



    def franke_function(self,x,y):
        """
        Calculate Franke's function from x and y values

        Args:
            x (array): x-values to use for computing th franke function.
            y (array): x-values to use for computing th franke function.

        Returns:
            array: array of same length as x, containing the computed function.

        """
        term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
        term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
        term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
        term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

        return term1 + term2 + term3 + term4

    def dataset_franke(self,n):
        """
        Create a data set using random values between 0 and 1 and calculate
        franke's function from those values'

        Args:
            n (int): total amount of data points.

        Returns:
            None.

        """
        self.x = np.random.rand(n)
        self.y = np.random.rand(n)
        noise = 0.1*np.random.randn(n)
        self.f = self.franke_function(self.x, self.y) + noise

        self.o2 = np.var(noise)
        self.n = n

    def data_setup(self,x,y,z):
        """
        create class variables from x,y and z, so they can be used in the class

        Args:
            x (array): x coordinate of dataset.
            y (array): y coordinate of dataset.
            z (array): data points

        Returns:
            None.

        """
        # to save visuals in correct folder
        self.terrain = True

        self.n = len(x)
        self.x ,self.y, self.f = x,y,z


    def design_matrix(self, deg):
        """
        create design matrix

        Args:
            deg (int): complexity of the model.

        Returns:
            X (matrix): design matrix of shape (n_datapoints, p_features).

        """
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
        """
        Split into training and testing data

        Args:
            X (matrix): Design matrix.
            f (array): Response variable
            scale (boolean, optional): if True scales the data. Defaults to True.

        Returns:
            None.

        """
        # Scaling Data
        if scale:
            X[:,1:] = (X[:,1:] - np.mean(X[:,1:], axis=0))/np.std(X[:,1:], axis=0)
            f = (f - np.mean(f))/np.std(f)
        # Splitting Data
        np.random.seed(42)
        self.X_train, self.X_test, self.f_train, self.f_test = train_test_split(X,f,test_size=0.2)


    def MSE(self,y,y_pred):
        """
        Calculate the mean squared error

        Args:
            y (array): Testing data set for y.
            y_pred (array): prediction for y.

        Returns:
            mse (float): Mean squared error.

        """
        mse = np.mean((y - y_pred)**2)
        return mse

    def OLS(self, X_train, X_test, f_train, lam=0):
        """
        Compute model with Ordinary Least Squared

        Args:
            X_train (matrix): Training data for design matrix.
            X_test (matrix): Testing data for design matrix.
            f_train (array):
            lam (float >=0, optional): set to zero in OLS. Defaults to 0.

        Raises:
            ValueError: if a lambda value lam>0 is used.

        Returns:
            f_tilde (array): prediciton for training data
            f_pred (array): prediction

        """
        if lam>0:
            raise ValueError('You are trying to use OLS with a lambda value greater than 0')


        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ f_train
        self.beta_OLS = beta # Store this beta in class (for confidence interval)
        f_tilde = X_train @ beta
        f_pred = X_test @ beta

        #print("filde:", f_tilde.shape)
        #print("fpred: ", f_pred.shape)

        return f_tilde, f_pred

    def bootstrap(self, X_train, X_test, f_train, trials, method, lam):
        """
        Use bootrap resampling to campute prediction and MSE

        Args:
            X_train (matrix): Training data from the split method.
            X_test (matrix): Testing data from the split method.
            f_train (array): Training data from the split method.
            trials (int): Number of bootstraps.
            method (function): which linear reression method to use, eg OLS.
            lam (float): which lambda value to use if ridge/lasso. if OLS set lam =0

        Returns:
            z_pred (matrix):  matrix of predictions for all bootstraps

        """
        mse = np.zeros(trials)

        z_pred = np.zeros((self.f_test.shape[0], trials))
        for i in range(trials):
            X_new, f_new = resample(X_train, f_train)
            z_pred[:,i] = method(X_new, X_test, f_new, lam)[1]
            mse[i] = self.MSE( self.f_test, z_pred[:,i] )
        return z_pred

    def beta_confidence(self, beta, o2, X_train,n):
        """
        calculate beta confidence interval

        Args:
            beta (array): beta values
            o2 (matrix): variance of error
            X_train (matrix): DESCRIPTION.
            n (int): DESCRIPTION.

        Returns:
            var_beta (array): beta variance values.

        """
        # Calculating variance
        var_beta_matrix = o2 * np.linalg.pinv(X_train.T @ X_train)
        var_beta = np.diagonal(var_beta_matrix)

        # Calculating confidence interval
        self.confidence = np.zeros((var_beta.shape[0], 2))
        self.confidence[:,0] = beta - 1.96 * 1/np.sqrt(n) * var_beta
        self.confidence[:,1] = beta + 1.96 * 1/np.sqrt(n) * var_beta
        return var_beta

    def ridge(self, X_train, X_test, f_train, lam):
        """
        make model using ridge

        Args:
            X_train (matrix): training data for X
            X_test (matrix): Testing data for X
            f_train (array): Training data for f
            lam (float): lambda value for lasso. Decides the penalty.

        Returns:
            f_tilde (array): prediction for training data
            f_pred (array): prediction

        """
        beta = np.linalg.pinv(X_train.T @ X_train + np.identity(len(X_train[0,:]))*lam) @ X_train.T @ f_train
        f_tilde = X_train @ beta
        f_pred = X_test @ beta

        self.beta_for_plot_ridge = beta

        return f_tilde, f_pred

    @ignore_warnings(category=ConvergenceWarning)
    def lasso(self,X_train, X_test, f_train, lam):
        """
        make model using lasso from sklearn

        Args:
            X_train (matrix): training data for X
            X_test (matrix): Testing data for X
            f_train (array): Training data for f
            lam (float): lambda value for lasso. Decides the penalty.

        Returns:
            f_tilde (array): prediction for training data
            f_pred (array): prediction

        """

        lassoreg = Lasso(alpha = lam)
        lassoreg.fit(X_train,f_train)
        f_tilde = lassoreg.predict(X_train)
        f_pred = lassoreg.predict(X_test)

        self.beta_for_plot_lasso = lassoreg.coef_

        return f_tilde, f_pred


    def k_fold(self,X,k, method, lam = None):
        """
        Evaluate mode using K-Fold cross validation

        Args:
            X (matrix): Design matrix.
            k (int): number of folds.
            method (function): which method to use, eg. Ridge.
            lam (TYPE, optional): which lambda value to use if choose ridge or
                    lasso. Defaults to None.

        Returns:
            float: Test Error.

        """

        #scaling data
        X[:,1:] = (X[:,1:] - np.mean(X[:,1:], axis=0))/np.std(X[:,1:])
        f = (self.f - np.mean(self.f))/np.std(self.f)

        #splitting data

        #removing some values if the array cannot be devided into k equal parts
        number = (len(f)//k)
        X = X[:(number*k),:]
        f = f[:(number*k)]

        X_lst1 = np.split(X,k)
        f_lst1 = np.split(f,k)

        X_lst = []
        f_lst = []

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

            for j in range(len(X_train)):
                for k in range(len(X_train[j])):
                    f_train_lst.append(f_train[j][k])
                    X_train_lst.append(X_train[j][k])

            f_train = np.array(f_train_lst)
            X_train = np.array(X_train_lst)

            f_tilde, z_pred = method(X_train,X_test,f_train, lam = lam)

            test_err_arr[i] = np.mean( (z_pred - f_test)**2 )
            train_err_arr[i] = np.mean( (f_tilde - f_train)**2 )


        return np.mean(test_err_arr)


    def heatmap(self, data, title, ticks_x=None, ticks_y=None, save = False, filename = None):
        """


        Args:
            data (matrix): data to plot
            title (string): title of plot
            ticks (list of strings, optional): tick to put on the x axis
                    . Defaults to None.
            save (boolean, optional): if True, saves the plot as a pdf
                                    . Defaults to False.
            filename (string, optional): filename without file extension
                                        . Defaults to None.

        Returns:
            None.

        """

        if ticks_x:
            sb.heatmap(data, cmap='coolwarm',
                   square = True, xticklabels = ticks_x, yticklabels = ticks_y).invert_yaxis()
        else:
            sb.heatmap(data, cmap='coolwarm',
                   square = True).invert_yaxis()

        plt.xlabel(r'$log10(\lambda)$', fontsize='16')
        plt.ylabel('Complexity', fontsize='16')
        plt.title(title, fontsize='16')
        plt.tick_params(labelsize='12')
        if save and self.terrain:
            print("Terrainsave!")
            plt.savefig('../visuals/Terrain_data/terrain_' + filename + '.pdf')
        elif save and not self.terrain:
            print("Frankesave!")
            plt.savefig('../visuals/Frankes_function/' + filename + '.pdf')
        plt.show()

    def single_plot(self, x,y, label_x, label_y, func_label, title, save = False, filename = None):
        """
        viusualise data in a plot

        Args:
            x (list of arrays): First element of list is the array used to plot
                        the x-values of function 1, the second element function 2 etc.
            y (list of arrays): Same as x but with y-values.
            label_x (string): Label for x axis.
            label_y (string): Label for y axis.
            func_label (list of strings): Labels to go in the legend.
            title (string): Title
            save (boolean, optional): if true, saves the plot to folder as pdf.
                            Defaults to False.
            filename (string, optional): filename without file extension
                                        . Defaults to None.

        Returns:
            None.

        """
        plt.style.use('seaborn-whitegrid')
        plt.title(title, fontsize='16')

        for i in range(len(x)):
            plt.plot(x[i], y[i], label = func_label[i])

        plt.xlabel(label_x, fontsize='16')
        plt.ylabel(label_y, fontsize='16')
        plt.tick_params(labelsize='12')
        plt.legend(fontsize='12')
        if save and self.terrain:
            plt.savefig('../visuals/Terrain_data/terrain_' + filename + '.pdf')
        elif save and self.terrain == False:
            plt.savefig('../visuals/Frankes_function/' + filename + '.pdf')
        plt.show()


def coef_plot(deg, n, lam_lst):
    """
    Plot coefficients for lasso and ridge

    Args:
        deg (int): Complexity: decides number of coefficients.
        n (int): Data points.
        lam_lst (array): lambda values to loop through.

    Returns:
        None.

    """
    reg = Regression()
    reg.dataset_franke(n)
    X = reg.design_matrix(deg)
    reg.split(X, reg.f)

    ridge_coefs = []
    lasso_coefs = []

    for lamb in lam_lst:
        reg.ridge(reg.X_train, reg.X_test, reg.f_train, lamb)
        ridge_coefs.append(reg.beta_for_plot_ridge)

        reg.lasso(reg.X_train, reg.X_test, reg.f_train, lamb)
        lasso_coefs.append(reg.beta_for_plot_lasso)

    lam_lst = np.log10(lam_lst)
    fig, axs = plt.subplots(2, sharey=True)
    fig.suptitle(r'$\beta$ for increasing $\lambda$ values',fontsize='16')
    axs[0].plot(lam_lst, ridge_coefs)
    axs[0].set_title('Ridge', fontsize='14')
    axs[1].plot(lam_lst, lasso_coefs)
    axs[1].set_title('Lasso', fontsize='14')
    fig.text(0.04, 0.5, r'$\beta$', ha='center',fontsize='16')

    plt.xlabel(r'$\lambda$',fontsize='16')
    plt.savefig('../visuals/coefplot_ridge_lasso.pdf')
    plt.show()
