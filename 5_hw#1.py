#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:21:22 2017

@author: kathy
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import leastsq

def load():
    """
        Load the data
    """
    load_data = loadmat('5_X.mat')
    load_target = loadmat('5_T.mat')
    data = load_data['X']
    target = load_target['T']
    train_x_1, train_x_2, train_x_3 = np.asarray(data[0:40]), np.asarray(data[50:90]), np.asarray(data[100:140])
    train_x = np.vstack((train_x_1, train_x_2, train_x_3))
    train_t_1, train_t_2, train_t_3 = np.asarray(target[0:40]), np.asarray(target[50:90]), np.asarray(target[100: 140])
    train_t = np.vstack((train_t_1, train_t_2, train_t_3))
    test_x_1, test_x_2, test_x_3 = np.asarray(data[40:50]), np.asarray(data[90:100]), np.asarray(data[140:150])
    test_x = np.vstack((test_x_1,test_x_2, test_x_3))
    test_t_1, test_t_2, test_t_3 = np.asarray(target[40:50]), np.asarray(target[90:100]), np.asarray(target[140:150])
    test_t = np.vstack((test_t_1, test_t_2, test_t_3))
    
    return train_x, train_t, test_x, test_t

def sep_data():
    load()
    train_x_col1, train_x_col2, train_x_col3, train_x_col4 = np.hsplit(train_x, 4)[0], np.hsplit(train_x, 4)[1], np.hsplit(train_x, 4)[2], np.hsplit(train_x, 4)[3]
    test_x_col1, test_x_col2, test_x_col3, test_x_col4 = np.hsplit(test_x, 4)[0], np.hsplit(test_x, 4)[1], np.hsplit(test_x, 4)[2], np.hsplit(test_x, 4)[3]
    
    return train_x_col1, train_x_col2, train_x_col3, train_x_col4, test_x_col1, test_x_col2, test_x_col3, test_x_col4

def poly(M):
    """
        make basis function
    """
    train_x_col1, train_x_col2, train_x_col3, train_x_col4, test_x_col1, test_x_col2, test_x_col3, test_x_col4 = sep_data()
    x_basis_train = np.ones((train_x.shape[0], 1))
    x_basis_test = np.ones((test_x.shape[0], 1))
    # basis function
    x_basis_train = np.hstack((x_basis_train, train_x_col1, train_x_col2, train_x_col3, train_x_col4))
    x_basis_test = np.hstack((x_basis_test, test_x_col1, test_x_col2, test_x_col3, test_x_col4))
    
    # train
    train_x_11 = train_x_col1 * train_x_col1
    train_x_12 = train_x_col1 * train_x_col2
    train_x_13 = train_x_col1 * train_x_col3
    train_x_14 = train_x_col1 * train_x_col4
    train_x_22 = train_x_col2 * train_x_col2
    train_x_23 = train_x_col2 * train_x_col3
    train_x_24 = train_x_col2 * train_x_col4
    train_x_33 = train_x_col3 * train_x_col3
    train_x_34 = train_x_col3 * train_x_col4
    train_x_44 = train_x_col4 * train_x_col4
    # test
    test_x_11 = test_x_col1 * test_x_col1
    test_x_12 = test_x_col1 * test_x_col2
    test_x_13 = test_x_col1 * test_x_col3
    test_x_14 = test_x_col1 * test_x_col4
    test_x_22 = test_x_col2 * test_x_col2
    test_x_23 = test_x_col2 * test_x_col3
    test_x_24 = test_x_col2 * test_x_col4
    test_x_33 = test_x_col3 * test_x_col3
    test_x_34 = test_x_col3 * test_x_col4
    test_x_44 = test_x_col4 * test_x_col4
        
    if M == 2:
        # basis function
        x_basis_train = np.hstack((x_basis_train,\
                                   train_x_11, train_x_12, train_x_13, train_x_14,\
                                   train_x_12, train_x_22, train_x_23, train_x_24,\
                                   train_x_13, train_x_23, train_x_33, train_x_34,\
                                   train_x_14, train_x_24, train_x_34, train_x_44))
        x_basis_test = np.hstack((x_basis_test,\
                                   test_x_11, test_x_12, test_x_13, test_x_14,\
                                   test_x_12, test_x_22, test_x_23, test_x_24,\
                                   test_x_13, test_x_23, test_x_33,test_x_34,\
                                   test_x_14, test_x_24, test_x_34,test_x_44))
    
    return x_basis_train, x_basis_test

def getCurve(x, t):
    """
        W* = (X^T * X)^-1 * X^T * Y
    """
    return np.dot(np.dot(np.linalg.pinv(np.dot(x.T, x)), x.T), t)

def error(w, x, t):
    """
        error function
    """
    return 0.5* np.sum((np.dot(x, w) - t) ** 2)


def rms_error(e, N):
    """
        RMS error function
    """
    return math.sqrt(e/N)


def drawError(train_errror_list, test_error_list, title):
    M = [1, 2]
    plt.plot(M, train_errror_list, '-o')
    plt.plot(M, test_error_list, '-*')
    plt.title(title)
    plt.xlabel("M")
    plt.ylabel("ERMS")
    plt.show()

if __name__ == '__main__':
    train_x, train_t, test_x, test_t = load()
    
    # setting
    M = [1, 2]
    
    # storage
    weight = [] # for w
    train_error = [] # for tarin rms error
    test_error = []
    
    
    for order in M:
        train_x_hyper, test_x_hyper = poly(order)
        
        w = getCurve(train_x_hyper, train_t)
        weight.append(w)
        
        train_error.append(rms_error(error(w, train_x_hyper, train_t), train_x.shape[0]))
        test_error.append(rms_error(error(w, test_x_hyper, test_t), test_x.shape[0]))
    
    drawError(train_error, test_error, "RMS ERROR")
