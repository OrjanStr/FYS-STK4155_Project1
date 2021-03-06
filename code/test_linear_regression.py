#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from linear_regression import Regression

import unittest

class TestLinearRegression(unittest.TestCase):

        
    
    def test_DM(self):

        x = np.linspace(1,5,5)
        y = np.linspace(1,5,5)
        z = np.linspace(1,5,5)
        
        reg = Regression()
        reg.data_setup(x,y,z)
        actual = reg.design_matrix(3)
        
        expected = np.zeros((5,10))
        expected[0] = [  1,   1,   1,   1,   1,   1,   1,   1,   1,   1]
        expected[1] = [  1,   2,   4,   8,   2,   4,   8,   4,   8,   8]
        expected[2] = [  1,   3,   9,  27,   3,   9,  27,   9,  27,  27]
        expected[3] = [  1,   4,  16,  64,   4,  16,  64,  16,  64,  64]
        expected[4] = [  1,   5,  25, 125,   5,  25, 125,  25, 125, 125]

        #test dimensions of design martix
        self.assertEqual(actual.shape, expected.shape)
        
        # test values of design matrix
        np.testing.assert_array_equal(actual, expected)
       
        
    def test_beta(self):
        deg = 6
        reg = Regression()
        reg.dataset_franke(10)
        X = reg.design_matrix(deg)
        reg.split(X, reg.f)
        reg.OLS(reg.X_train, reg.X_test, reg.f_train)
        actual = reg.beta_OLS
        expected = (28,)
        
        #test dimension of beta
        np.testing.assert_array_equal(actual.shape, expected)
        
        
    def test_OLS(self):
        x = np.linspace(0,10,200)
        y = np.linspace(0,10,200)
        f = lambda x,y: 5*x*y**2  #function for 3.deg polynomial
        z = f(x,y)
        
        degrees = np.linspace(1,5,5, dtype = int)
        mse_lst= np.zeros(len(degrees))
        
        reg = Regression()
        reg.data_setup(x,y,z)
        for i, deg in enumerate(degrees):
            X = reg.design_matrix(deg)
            reg.split(X, reg.f)
            f_tilde, f_pred = reg.OLS(reg.X_train, reg.X_test, reg.f_train)
            mse_lst[i] = np.mean((f_pred - reg.f_test)**2)

        # test that OLS returns correct degree as best fit
        index = np.argmin(mse_lst)
        self.assertEqual(index, 2) # index 2 = 3rd degree 
        
    def test_ridge(self):
        x = np.linspace(0,10,200)
        y = np.linspace(0,10,200)
        f = lambda x,y: 5*x*y**2  #function for 3.deg polynomial
        z = f(x,y)
        
        degrees = np.linspace(1,5,5, dtype = int)
        mse_lst= np.zeros(len(degrees))
        
        reg = Regression()
        reg.data_setup(x,y,z)
        for i, deg in enumerate(degrees):
            X = reg.design_matrix(deg)
            reg.split(X, reg.f)
            f_tilde, f_pred = reg.ridge(reg.X_train, reg.X_test, reg.f_train, lam=0)
            mse_lst[i] = np.mean((f_pred - reg.f_test)**2)
            
        # test that Ridge returns correct degree as best fit
        index = np.argmin(mse_lst)
        self.assertEqual(index, 2) # index 2 = 3rd degree 

        
        
        
if __name__ == '__main__':

    unittest.main()