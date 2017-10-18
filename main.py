"""
Created on ???

@author: Kethy
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

M = 9

def sort(x, t):
    x = np.asarray(x)
    t = np.asarray(t)
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            if x[i] > x[j]:
                x[i], x[j] = x[j], x[i]
                t[i], t[j] = t[j], t[i]
    return x, t

def load():
    """
        Load the data
    """
    train_data = pd.read_csv("4_train.csv", sep = ",")
    test_data = pd.read_csv("4_test.csv", sep = ",")
    train_x, train_t = sort(train_data['x'], train_data['t'])
    test_x, test_t = sort(test_data['x'], test_data['t'])
    return train_x, train_t, test_x, test_t

def poly(x, M):
    """
        Do the basis function projection

        Return: (M * 20)
    """
    x_basis = None
    x = np.expand_dims(x, axis=0)
    for i in range(M):
        if type(x_basis) == type(None):
            x_basis = x
        else:
            x_basis = np.concatenate((x_basis, x ** i), axis=0)
    return x_basis.T

def getCurve(x, y):
    """
        W* = (X^T * X)^-1 * X^T * y
    """
    return np.dot(np.dot(np.linalg.pinv(np.dot(x.T, x)), x.T), y)

def getRegularizedCurve(x, y, _lambda):
    """
        W* = (X^T * X + lambda * I)^-1 * X^T * y
    """
    panelty = _lambda * np.ones([np.shape(np.dot(x.T, x))[0]])
    return np.dot(np.dot(np.linalg.pinv(np.dot(x.T, x) + panelty), x.T), y)

def error(w, x, t):
    """
        define error function
    """
    return 0.5 * np.sum((np.dot(x, w) - t) ** 2)

def drawCurve(train_x, train_x_hyper, train_y, test_x, test_x_hyper, test_y, w, title):
    plt.plot(train_x, train_y, '-o', label="train label")
    plt.plot(train_x, np.dot(train_x_hyper, w), '-o', label='train logit')
    plt.plot(test_x, test_y, '-o', label="test label")
    plt.plot(test_x, np.dot(test_x_hyper, w), '-o', label='test logit')
    plt.title(title)
    plt.legend()

def drawError(train_error_list, test_error_list, title):
    x = np.linspace(1, len(train_error_list), num=len(train_error_list))
    plt.plot(x, train_error_list, '-o', label='train error')
    plt.plot(x, test_error_list, '-o', label='test error')
    plt.title(title)
    plt.legend()

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load()

    # Initial object
    lse_train_error = []
    lse_test_error = []
    rlse_train_error = []
    rlse_test_error = []

    # --------------------------
    # Usual LSE
    # --------------------------
    print('<--  LSE -->')
    plt.figure(1)
    for i in range(1, M+1):
        # basis function projection
        train_x_hyper = poly(train_x, i)
        test_x_hyper = poly(test_x, i)

        # find curve
        curve = getCurve(train_x_hyper, train_y)

        # print error
        print('M: ', i, '\ttraining error: ', error(curve, train_x_hyper, train_y), '\ttesting error: ', error(curve, test_x_hyper, test_y))
        lse_train_error.append(error(curve, train_x_hyper, train_y))
        lse_test_error.append(error(curve, test_x_hyper, test_y))

        # Draw
        plt.subplot(3, math.ceil(M/3), i)
        drawCurve(train_x, train_x_hyper, train_y, test_x, test_x_hyper, test_y, curve, 'M='+str(i))

    # --------------------------
    # Regularized LSE
    # --------------------------
    print('<--  Regularized LSE (M=9)-->')
    plt.figure(2)
    lambd_list = [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
    for i in range(len(lambd_list)):
        # basis function projection
        train_x_hyper = poly(train_x, M)
        test_x_hyper = poly(test_x, M)

        # find curve
        curve = getRegularizedCurve(train_x_hyper, train_y, lambd_list[i])

        # print error
        print('lanbda: ', lambd_list[i], '\ttraining error: ', error(curve, train_x_hyper, train_y), '\ttesting error: ', error(curve, test_x_hyper, test_y))
        rlse_train_error.append(error(curve, train_x_hyper, train_y))
        rlse_test_error.append(error(curve, test_x_hyper, test_y))

        # Draw
        plt.subplot(3, math.ceil(len(lambd_list)/3), i+1)
        drawCurve(train_x, train_x_hyper, train_y, test_x, test_x_hyper, test_y, curve, 'lambda='+str(lambd_list[i]))

    # --------------------------
    # Draw error
    # --------------------------
    plt.figure(3)
    drawError(lse_train_error, lse_test_error, 'poly error')
    plt.figure(4)
    drawError(rlse_train_error, rlse_test_error, 'regularized error')
    plt.show()