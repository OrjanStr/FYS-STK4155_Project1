#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


from frankfunc import Regression
import unittest

class TestFrankfunk(unittest.TestCase):

        
    
    def test_DM(self):
        x = np.linspace(1,5,5)
        y = np.linspace(1,5,5)
        z = np.linspace(1,5,5)
        
        reg = Regression()
        reg.data_setup(x,y,z)
        actual = reg.design_matrix(3)
        
        expected = np.zeros((5,10))
        expected[0] = [  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,]
        expected[1] = [  1,   2,   4,   8,   2,   4,   8,   4,   8,   8,]
        expected[2] = [  1,   3,   9,  27,   3,   9,  27,   9,  27,  27,]
        expected[3] = [  1,   4,  16,  64,   4,  16,  64,  16,  64,  64,]
        expected[4] = [  1,   5,  25, 125,   5,  25, 125,  25, 125, 125,]
    
        try:
            self.assertEqual(actual.shape, expected.shape)
        except AssertionError:
            print ('The dimentions of the design matrix are incorrect\nexpected: ',expected.shape, '\nactual:    ',actual.shape)
            
        try:
            np.testing.assert_array_equal(actual, expected)
        except AssertionError:
            print ('The values of the design matrix are incorrect\nexpected:\n ',expected, '\nactual:\n    ',actual)
        
        
    def test_beta(self):
        deg = 6
        reg = Regression()
        reg.dataset_franke(10)
        X = reg.design_matrix(deg)
        X_train, X_test, f_train, f_test = reg.split(X, reg.f)
        reg.OLS(X_train, X_test, f_train)
        actual = reg.beta_OLS
        expected = (28,)
        try:
            np.testing.assert_array_equal(actual.shape, expected)
        except AssertionError:
            print ('The dimentions of the design matrix are incorrect\nexpected: ',expected, '\nactual:    ',actual.shape)
        
        
TFF = TestFrankfunk()
TFF.test_DM()
TFF.test_beta()