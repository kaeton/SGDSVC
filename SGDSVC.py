#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:set fileencoding=utf-8:

import numpy as np

class SGDSVC(object):
    def __init__(self,
                 kernel="rbf", lmd=1e-1, gamma=0.1, bias=1.0, max_iter=100):
        if kernel not in self.__kernel_dict:
            print(kernel + " kernel does not exist!\nUse rbf kernel.")
            kernel = "rbf"
        if kernel == "rbf":
            def kernel_func(x, y):
                return self.__kernel_dict[kernel](x, y, gamma=gamma)
        else:
            kernel_func = self.__kernel_dict[kernel]
        self.kernel = kernel_func
        self.lmd = lmd
        self.bias = bias
        self.max_iter = max_iter

    def __linear_kernel(x, y):
        return np.dot(x, y)

    def __get_alfavalues(self):
        alpha = self.alpha
        #print("get alfa values function")
        #print(alpha)
        length = [x[0] for x in [i for i in alpha]]
        return(np.count_nonzero(length))

    def __gaussian_kernel(x, y, gamma):
        #print("x:"+str(np.shape(x)))
        #print("y:"+str(np.shape(y)))
        diff = x - y
        return np.exp(-gamma * np.dot(diff, diff))

    __kernel_dict = {"linear": __linear_kernel, "rbf": __gaussian_kernel}

    def fit(self, X, y):
        def update_alpha(alpha, t):
            data_size, feature_size = np.shape(self.X_with_bias)
            new_alpha = np.copy(alpha)
            it = np.random.randint(low=0, high=data_size)
            x_it = self.X_with_bias[it]
            y_it = self.y[it]
            if (y_it *
                (1. / (self.lmd * t)) *
                sum([alpha_j * y_it * self.kernel(x_it, x_j)
                     for x_j, alpha_j in zip(self.X_with_bias, alpha)])) < 1.:
                new_alpha[it] += 1
            return new_alpha

        print(str(np.shape(X)))
        self.X_with_bias = np.c_[X, np.ones((np.shape(X)[0])) * self.bias]
        self.y = y
        alpha = np.zeros((np.shape(self.X_with_bias)[0], 1))
        for t in range(1, self.max_iter + 1):
            alpha = update_alpha(alpha, t)

        #print(alpha)
        #print(self.alpha)
        self.alpha = alpha

    def decision_function(self, X):
        print(str(np.shape(X)))
        X_with_bias = np.c_[X, np.ones((np.shape(X)[0])) * self.bias]
        y_score = [(1. / (self.lmd * self.max_iter)) *
                   sum([alpha_j * y_j * self.kernel(x_j, x)
                        for (x_j, y_j, alpha_j) in zip(
                        self.X_with_bias, self.y, self.alpha)])
                   for x in X_with_bias]
        return np.array(y_score)

    def predict(self, X):
        y_score = self.decision_function(X)
        y_predict = map(lambda s: 1 if s >= 0. else -1, y_score)
        return y_predict