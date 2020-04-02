"""
Module provides functions for manipulation and testing of data for both parts
of project:

1) Time series analysis on all monthly rentals
2) Time series analysis on broken down data between monthly member rentals and
   monthly casual rentals
"""
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import pandas as pd
import plots as p
import functions as f


def test_stationarity(data, window, title):
    """
    Creates stationarity plot, returns results of Dickey-Fuller test
    """
    p.stationarity_plot(data, window, title)
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(data, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic',
                                'p-value',
                                '#Lags Used',
                                'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def stationarity_test(data, var='count', diff=False, s_diff=False):
    """
    Tests stationarity and autocorrelation of time series
    """
    data2 = data.copy()
    if diff and s_diff:
        data2 = f.order_difference(data2, var=var, diff=True, s_diff=True)
        g_title = "Seasonal Difference of First Order Difference"
    elif diff:
        data2 = f.order_difference(data2, var=var, diff=True, s_diff=False)
        g_title = "First Order Difference"
    elif s_diff:
        data2 = f.order_difference(data2, var=var, diff=False, s_diff=True)
        g_title = "Seasonal Difference"
    else:
        g_title = "Original Time Series"
    
    test_stationarity(data2[var], window=12, title=g_title)

    #p.acf_pacf(data)





def sarima(data, orders, s_orders, var='count', test=False):
    """
    Evaluates SARIMA models for all combinations of orders and seasonal orders
    to be tested. Plots model output on top of training and test data as visual
    aid, prints training and testing mean squared error for each model. This
    function is for part 1 of the project (all monthly rentals).
    """
    best = 0
    best_o = []
    best_so = []

    for o_val in orders:
        for s_val in s_orders:
            model = sm.tsa.statespace.SARIMAX(data[var],
                                              order=(o_val[0], o_val[1], o_val[2]),
                                              seasonal_order=(
                                                  s_val[0], s_val[1], s_val[2],
                                                  s_val[3])).fit()
            print(f'{o_val}, {s_val}:')
            mse = compare_mse(model, data, return_val=True, var=var)
            if best==0 or best>mse:
                best = mse
                best_o = o_val
                best_so = s_val
                best_model = model
    print(f'Best MSE: {(best):.2e}')
    print(f'Best Model: {best_o}, {best_so}')
            
    p.prediction_plot(best_model, data, best_o, best_so, test=False, var=var)


def sarima_breakdown(train_df, test_df):
    """
    Evaluates SARIMA models for all combinations of orders and seasonal orders
    to be tested for both monthly member rentals and monthly casual rentals.
    Plots model output on top of training and test data as visual
    aid, prints training and testing mean squared error for each model. This
    function is for part 2 of the project (broken down monthly rentals between
    member and casual).
    """
    orders = [(2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 1, 3)]
    seasonal_orders = [(0, 1, 0, 12), (1, 1, 0, 12), (0, 1, 1, 12)]
    for o_val in orders:
        for s_o in seasonal_orders:
            print(f'Orders: {o_val}. Seasonal Orders: {s_o}')
            model = sm.tsa.statespace.SARIMAX(train_df['member'],
                                              order=(o_val[0], o_val[1],
                                                     o_val[2]),
                                              seasonal_order=(s_o[0],
                                                              s_o[1],
                                                              s_o[2],
                                                              s_o[3])).fit()
            print(model.summary())
            _, _ = compare_mse_breakdown(model, train_df,
                                         test_df, 'member')
            p.prediction_plot_breakdown(model, train_df, test_df, 'member',
                                            o_val[0], o_val[1], o_val[2],
                                            s_o[0],
                                            s_o[1], s_o[2], s_o[3])

            model = sm.tsa.statespace.SARIMAX(train_df['casual'],
                                              order=(o_val[0], o_val[1],
                                                     o_val[2]),
                                              seasonal_order=(s_o[0], s_o[1],
                                                              s_o[2],
                                                              s_o[3])).fit()
            print(model.summary())
            _, _ = compare_mse_breakdown(model, train_df,
                                         test_df, 'casual')
            p.prediction_plot_breakdown(model, train_df, test_df, 'casual',
                                            o_val[0], o_val[1], o_val[2],
                                            s_o[0],
                                            s_o[1], s_o[2], s_o[3])


def compare_mse(sarima_model, data, test=False, return_val=False, var='count'):
    """
    Calculates mean squared errors (MSE) for the training data
    for a given model so that mse can be compared across models. This function
    is for part 1 of the project (all monthly rentals).
    """
    if test:
        data, test_df = f.train_split(data)
    
    predict_train = sarima_model.predict(start=0, end=(len(data)))
    train_mse = mean_squared_error(data[var], predict_train[:-1])
    print('Training MSE: ', '{:.2e}'.format(train_mse))
    
    if test: 
        predict_test = sarima_model.predict(start=(len(data)),
                                            end=(len(data) +
                                                 len(test_df)))
        test_mse = mean_squared_error(test_df[var], predict_test[:-1])
        print('Testing MSE: ', '{:.2e}'.format(test_mse))
    if return_val:
        return float(train_mse)       


def compare_mse_breakdown(sarima_model, training_set, testing_set, kind):
    """
    Calculates mean squared errors (MSE) for both the training and testing data
    for a given model so that mse can be compared across models. This function
    is for part 2 of the project (broken down monthly rentals between member
    and casual).
    """
    predict_train = sarima_model.predict(start=0, end=(len(training_set)))
    predict_test = sarima_model.predict(start=(len(training_set)),
                                        end=(len(training_set) +
                                             len(testing_set)))
    train_mse = mean_squared_error(training_set[kind], predict_train[:-1])
    test_mse = mean_squared_error(testing_set[kind], predict_test[:-1])
    print('Training MSE: ', '{:.2e}'.format(train_mse))
    print('Testing MSE: ', '{:.2e}'.format(test_mse))
    return train_mse, test_mse
