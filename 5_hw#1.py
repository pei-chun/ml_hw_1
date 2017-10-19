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

def poly(w, x1, x2, x3, x4, M):
    """
        A general polynomial form
    """
    
    if M == 1:
        w0, w1, w2, w3, w4 = w
        func = w0 + w1*x1 + w2*x2 + w3*x2 + w4*x4
    elif M == 2:
        w0, w1, w2, w3, w4,\
        w11, w12, w13, w14,\
        w21, w22, w23, w24,\
        w31, w32, w33, w34,\
        w41, w42, w43, w44 = w
        func = w0 + w1*x1 + w2*x2 + w3*x3 + w4*x4\
        + w11*x1*x1 + w12*x1*x2 + w13*x1*x3 + w14*x1*x4\
        + w21*x2*x1 + w22*x2*x2 + w23*x2*x3 + w24*x2*x4\
        + w31*x3*x1 + w32*x3*x2 + w33*x3*x3 + w34*x3*x4\
        + w41*x4*x1 + w42*x4*x2 + w43*x4*x3 + w44*x4*x4
    return func

def error(w, x1, x2, x3, x4, M, t):
    """
        error function
    """
    return poly(w, x1, x2, x3, x4, M) - t

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
    train_x_col1, train_x_col2, train_x_col3, train_x_col4, test_x_col1, test_x_col2, test_x_col3, test_x_col4 = sep_data()
    # setting
    M = [1, 2]
    w_test = [[0]*5, [0]*21]
    # storage
    weight = [] # for w
    train_error = [] # for tarin rms error
    test_error = []
    
    
    for order in M:
        w = leastsq(error, w_test[order], args=(train_x_col1, train_x_col2, train_x_col3, train_x_col4, order, train_t))
        weight.append(w[0])
        
        train_error.append(rms_error(error(w[0], train_x, D, train_t), train_x.shape[0]))
        test_error.append(rms_error(error(w[0], test_x, D, test_t), test_x.shape[0]))
    
    drawError(train_error, test_error, "RMS ERROR")
