#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 22:49:28 2017

@author: kathy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import leastsq

"""
input data from data set
"""
train_data = pd.read_csv("~/Desktop/4_train.csv", sep = ",")
test_data = pd.read_csv("~/Desktop/4_test.csv", sep = ",")

x_train = train_data["x"]
y_train = train_data["t"]
x_test = test_data["x"]
y_test = test_data["t"]

"""
setting M
"""
M = list(range(1,10))


"""
define polynomial function
"""
def poly(w, x, M):
    if M == 1:
        w0, w1 = w
        func = w0 + w1*x
    elif M == 2:
        w0, w1, w2 = w
        func = w0 + w1*x + w2*(x**2)
    elif M == 3:
        w0, w1, w2, w3 = w
        func = w0 + w1*x + w2*(x**2) + w3*(x**3)
    elif M == 4:
        w0, w1, w2, w3, w4 = w
        func = w0 + w1*x + w2*(x**2) + w3*(x**3) + w4*(x**4)
    elif M == 5:
        w0, w1, w2, w3, w4, w5 = w
        func = w0 + w1*x + w2*(x**2) + w3*(x**3) + w4*(x**4) + w5*(x**5)
    elif M == 6:
        w0, w1, w2, w3, w4, w5, w6 = w
        func = w0 + w1*x + w2*(x**2) + w3*(x**3) + w4*(x**4) + w5*(x**5) + w6*(x**6)
    elif M == 7:
        w0, w1, w2, w3, w4, w5, w6, w7 = w
        func = w0 + w1*x + w2*(x**2) + w3*(x**3) + w4*(x**4) + w5*(x**5) + w6*(x**6) + w7*(x**7)
    elif M == 8:
        w0, w1, w2, w3, w4, w5, w6, w7, w8 = w
        func = w0 + w1*x + w2*(x**2) + w3*(x**3) + w4*(x**4) + w5*(x**5) + w6*(x**6) + w7*(x**7) + w8*(x**8)
    else:
        w0, w1, w2, w3, w4, w5, w6, w7, w8, w9 = w
        func = w0 + w1*x + w2*(x**2) + w3*(x**3) + w4*(x**4) + w5*(x**5) + w6*(x**6) + w7*(x**7) + w8*(x**8) + w9*(x**9)
    return func

"""
define error function
"""
def error(w, x, M, t):
    return poly(w, x, M) - t

"""
define lambda
"""


"""
calculate w of M form 1 to 9
"""
parameter = [] #store all the parameters from different M
error_rms_train = [] #store the rms error of training data
error_rms_test = [] #store the rms error of testing data
error_train = [] #store error function of different M
error_test = [] #store error function of different M

for order in M:
    w_test = [0] * (order + 1)
    
    curve = leastsq(error, w_test, args=(x_train, order, y_train))
    # calculate the minimum least square error to find all the w    
    parameter.insert(order-1, curve[0]) #store w
    print ("M = ", order)
    print ('w=', parameter[order-1])
    
    #calculate error function
    E_train = 0.5 * sum(error(curve[0], x_train, order, y_train)**2)
    E_test = 0.5 * sum(error(curve[0], x_test, order, y_test)**2)
    
    ERMS_train = math.sqrt((2*E_train)/x_train.shape[0])
    #print ("traing error: ", ERMS_train)
    ERMS_test = math.sqrt((2*E_test)/x_test.shape[0])
    #print ("testing error: ", ERMS_test)
    
    # calculate Error and Erms
    error_train.append(E_train)
    error_test.append(E_test)
    error_rms_train.append(ERMS_train)
    error_rms_test.append(ERMS_test)
    # store Erms
    
    """
    plot training data in 'o'
    plot polynomial curve
    """
    plt.plot(x_train, y_train, 'o')   
    test_curve = np.linspace(0, 7, 100)
    plt.plot(test_curve, poly(curve[0], test_curve, order))
    #plt.plot(x, poly(curve[0], x, order),'*')
    """
    some details of plot setting
    """
    plt.title("training data set (M = %d)"%order)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show() # call plt.show() to make graphics appear.
    
"""
plot error
"""
plt.plot(M, error_rms_train, '-o')
plt.plot(M, error_rms_test, '-*')

plt.title("root-mean-square error")
plt.xlabel("M")
plt.ylabel("ERMS")
plt.show()

"""----------Regularized----------"""
"""
define regularized error function
"""
def re_error(w, x, M, lam, t):
    return 0.5*((poly(w, x, M))
"""
plot regularized error
"""
plt.plot(lam, Re_error_train)
plt.plot(lam, Re_error_test)

plt.title("regularization")
plt.xlabel("ln lambda")
plt.ylabel("ERMS")
plt.show()
