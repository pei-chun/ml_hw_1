#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 19:39:01 2017

@author: kathy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def load():
    """
        Load the data
    """
    train_data = pd.read_csv("4_train.csv", sep = ",")
    test_data = pd.read_csv("4_test.csv", sep = ",")
    train_x, train_t = np.asarray(train_data['x']), np.asarray(train_data['t'])
    test_x, test_t = np.asarray(test_data['x']), np.asarray(test_data['t'])
    
    train_x, train_t = np.expand_dims(train_x, axis = 1), np.expand_dims(train_t, axis = 1)
    test_x, test_t = np.expand_dims(test_x, axis = 1), np.expand_dims(test_t, axis = 1)
    
    return train_x, train_t, test_x, test_t

def poly(x, M):
    """
        make basis function
    """
    x_basis = np.ones((x.shape[0], 1))
    
    for order in list(range(M)):
        x_basis_order = x ** (order+1)
        x_basis = np.hstack((x_basis, x_basis_order))
    
    return x_basis

def getCurve(x, y):
    """
        W* = (X^T * X)^-1 * X^T * y
    """
    return np.dot(np.dot(np.linalg.pinv(np.dot(x.T, x)), x.T), y)

def getRegularizedCurve(x, y, _lambda):
    """
        W* = (X^T * X + lambda * I)^-1 * X^T * y
    """
    panelty = _lambda * np.eye(np.shape(np.dot(x.T, x))[0])
    return np.dot(np.dot(np.linalg.pinv(np.dot(x.T, x) + panelty), x.T), y)

def error(w, x, t):
    """
        define error function
    """
    return 0.5 * np.sum((np.dot(x, w) - t) ** 2)

def rms_error(e, N):
    """
        RMS error function
    """
    return math.sqrt(2*e/N)

def drawError(x ,train_errror_list, test_error_list, title):
    plt.plot(x, train_errror_list, '-o')
    plt.plot(x, test_error_list, '-*')
    plt.title(title)
    plt.ylabel("ERMS")
    plt.show()

def drawReError(x ,train_errror_list, test_error_list, title):
    plt.plot(x, train_errror_list)
    plt.plot(x, test_error_list)
    plt.title(title)
    plt.ylabel("ERMS")
    plt.show()
    
if __name__ == '__main__':
    train_x, train_t, test_x, test_t = load()
    
    print('<--RMS Error form M = 1 to M = 9-->')
    # setting
    M = list(range(1, 10))
    
    #storage
    weight = []
    
    train_error = [] # for train rms error
    test_error = [] # for test rms error
    
    for order in M:
        train_x_hyper = poly(train_x, order)
        test_x_hyper = poly(test_x, order)
        
        w = getCurve(train_x_hyper, train_t)
        weight.append(w)
        
        train_error.append(rms_error(error(w, train_x_hyper, train_t), train_x.shape[0]))
        test_error.append(rms_error(error(w, test_x_hyper, test_t), test_x.shape[0]))
        
    drawError(M, train_error, test_error, "RMS ERRROR")
    
    print('<--Regularzed RMS Erro for M = 9-->')
    # setting
    ln_lambda = np.linspace(-20, 2.5, 100)
    _lambda = np.exp(ln_lambda)
    
    # storage
    Re_weight = []
    
    Re_train_error = []
    Re_test_error = []
    
    for i in _lambda:
        w = getRegularizedCurve(train_x_hyper, train_t, i)
        Re_weight.append(w)
        
        Re_train_error.append(rms_error(error(w, train_x_hyper, train_t), train_x.shape[0]))
        Re_test_error.append(rms_error(error(w, test_x_hyper, test_t), test_x.shape[0]))
        
    drawReError(ln_lambda, Re_train_error, Re_test_error, "RMS ERRROR")
