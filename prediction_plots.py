#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 20:25:43 2019

@author: marissaeppes
"""

import matplotlib.pyplot as plt
import pandas as pd


def prediction_plot(automodel, training_set, testing_set):
    
    if automodel.seasonal_order[3] == 52:
        freq ='Week'
        lw = 2
    elif automodel.seasonal_order[3] == 12:
        freq = 'Month'
        lw = 4
    else:
        pass
    
    order = str(automodel.order)
    seasonal_order = str(automodel.seasonal_order)
    
    predict_train = automodel.predict_in_sample()
    predict_test = automodel.predict(len(testing_set))
    
    fig = plt.figure(figsize = (20,20))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    
    ax1.plot(training_set, lw = lw, color = 'mediumturquoise')
    ax1.plot(testing_set, lw = lw, color = 'blue')
    ax1.plot(training_set.index,predict_train, lw = lw, color = 'magenta')
    
    ax2.plot(training_set, lw = lw, color = 'mediumturquoise')
    ax2.plot(testing_set, lw = lw, color = 'blue')
    ax2.plot(testing_set.index,predict_test, lw = lw, color = 'orange')
    
    ax1.set_xlabel('Date', fontsize = 20)
    ax1.set_ylabel('Count', fontsize = 20)
    
    ax2.set_xlabel('Date', fontsize = 20)
    ax2.set_ylabel('Count', fontsize = 20)
    
    ax1.set_title(f'Number of Bike Rentals per {freq} Time Series, SARIMA {order}{seasonal_order}', fontsize = 30)
    ax2.set_title(f'Number of Bike Rentals per {freq} Time Series, SARIMA {order}{seasonal_order}', fontsize = 30)
    
    ax1.legend(['Train','Test', 'Model Prediction on Training Set'],prop={'size': 24})
    ax2.legend(['Train','Test', 'Model Prediction on Testing Set'],prop={'size': 24})
    
def forecast_plot(automodel, master, n_forecast):
    
    if automodel.seasonal_order[3] == 52:
        freq ='Week'
        lw = 2
    elif automodel.seasonal_order[3] == 12:
        freq = 'Month'
        lw = 4
    else:
        pass
    
    order = str(automodel.order)
    seasonal_order = str(automodel.seasonal_order)
    
    fig = plt.figure(figsize = (20,10))
    plt.plot(master, lw = lw, color = 'mediumorchid')
    
    
    diff = len(master) - len(automodel.predict_in_sample())
    predict_next = automodel.predict(diff + n_forecast)
    forecast = predict_next[-n_forecast:]
    
    forecast_dates = pd.date_range(master.index[-1], freq = 'm', periods=n_forecast + 1).tolist()[1:]

    plt.plot(forecast_dates, forecast, lw = 6, color = 'indigo')
    
    plt.xlabel('Date', fontsize = 20)
    plt.ylabel('Count', fontsize = 20)
    plt.legend(['Data','Prediction'],prop={'size': 24})
    plt.title(f'{n_forecast}-{freq} Prediction on Bike Rentals Using SARIMA {order}{seasonal_order}',fontsize = 30)
    
    
    
        
    
    
    