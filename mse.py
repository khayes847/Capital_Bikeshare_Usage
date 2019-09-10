#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 19:04:22 2019

@author: marissaeppes
"""

from sklearn.metrics import mean_squared_error
import numpy as np



def compare_mse(sarima_model, training_set, testing_set):
    predict_train = sarima_model.predict(start = 0, end = len(training_set))
    predict_test = sarima_model.predict(start = len(training_set), end = 
                                        len(training_set) + len(testing_set))
    
    train_mse = mean_squared_error(training_set['count'], predict_train[:-1])
    test_mse = mean_squared_error(testing_set['count'], predict_test[:-1])
    
    print('Training MSE: ', "{:.2e}".format(train_mse))
    print('Testing MSE: ', "{:.2e}".format(test_mse))
    
    return train_mse, test_mse

def compare_mse_exog(sarima_model, training_df, testing_df):
    exog_train = np.array([training_df['avg_temp'], training_df['avg_precips']]).T
    exog_test = np.array([testing_df['avg_temp'], testing_df['avg_precips']]).T
    predict_train = sarima_model.predict(exog = exog_train)
    predict_test = sarima_model.predict(start = len(training_df), end = 
                                        len(training_df) + len(testing_df) - 1,
                                        exog = exog_test)
    
    train_mse = mean_squared_error(training_df['count'], predict_train)
    test_mse = mean_squared_error(testing_df['count'], predict_test)
    
    print('Training MSE: ', "{:.2e}".format(train_mse))
    print('Testing MSE: ', "{:.2e}".format(test_mse))
    
    return train_mse, test_mse


def compare_mse_auto(autosarima_model, training_set, testing_set):
    predict_train = autosarima_model.predict_in_sample()
    predict_test = autosarima_model.predict(len(testing_set))
    
    train_mse = mean_squared_error(training_set['count'], predict_train)
    test_mse = mean_squared_error(testing_set['count'], predict_test)
    
    print('Training MSE: ', "{:.2e}".format(train_mse))
    print('Testing MSE: ', "{:.2e}".format(test_mse))
    
    return train_mse, test_mse
    
    
    