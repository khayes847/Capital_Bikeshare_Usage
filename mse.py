#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 19:04:22 2019

@author: marissaeppes
"""

from sklearn.metrics import mean_squared_error



def compare_mse(sarima_model, training_set, testing_set):
    predict_train = sarima_model.predict(start = 0, end = len(training_set))
    predict_test = sarima_model.predict(start = len(training_set), end = 
                                        len(training_set) + len(testing_set))
    
    train_mse = mean_squared_error(training_set['count'], predict_train[:-1])
    test_mse = mean_squared_error(testing_set['count'], predict_test[:-1])
    
    print('Training MSE: ', train_mse)
    print('Testing MSE: ', test_mse)
    
    return train_mse, test_mse


def compare_mse_auto(autosarima_model, training_set, testing_set):
    predict_train = autosarima_model.predict_in_sample()
    predict_test = autosarima_model.predict(len(testing_set))
    
    train_mse = mean_squared_error(training_set['count'], predict_train)
    test_mse = mean_squared_error(testing_set['count'], predict_test)
    
    print('Training MSE: ', "{:.2e}".format(train_mse))
    print('Testing MSE: ', "{:.2e}".format(test_mse))
    
    return train_mse, test_mse
    
    
    