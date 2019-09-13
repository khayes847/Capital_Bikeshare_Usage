"""Functions used for data manipulation"""
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import pandas as pd
import prediction_plots_final as pplot
import functions as f
import operator
from collections import OrderedDict


def test_stationarity(data, window, title):
    """Creates stationarity plot, returns results of Dickey-Fuller test"""
    pplot.stationarity_plot(data, window, title)
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


def stationarity_autocorrelation_test_original(data):
    """Tests stationarity and autocorrelarity of timeseries"""
    print('The purpose of this test is to determine the stationarity and '
          'autocorrelarity\n of the original timeseries.')
    test_stationarity(data['count'], window=12,
                      title="Original Timeseries")
    pplot.acf_pacf(data)


def sarima(train_m, test_m):
    """Runs Sarima test on member data sets according to orders"""
    orders = [(2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 1, 3)]
    seasonal_orders = [(0, 1, 0, 12), (1, 1, 0, 12), (0, 1, 1, 12)]
    mse_dict = {}
    for o in orders:
        for s_o in seasonal_orders:
            model = sm.tsa.statespace.SARIMAX(train_m['count'], order=(o[0], o[1], o[2]),
                                              seasonal_order=(s_o[0], s_o[1], s_o[2], s_o[3])).fit()
            print(model.summary())
            train_mse, test_mse = mse.compare_mse(model, train_m, test_m)
            pplot.prediction_plot(model, train_m, test_m,
                                  o[0], o[1], o[2], s_o[0], s_o[1], s_o[2], s_o[3])

            mse_dict[f'{o},{s_o}'] = {'Training MSE': "{:.2e}".format(
                train_mse), 'Testing MSE': "{:.2e}".format(test_mse)}
    return train_mse, test_mse


def sarima_members(train_df, test_df):
    """Runs Sarima test on member data sets according to orders"""
    orders = [(2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 1, 3)]
    seasonal_orders = [(0, 1, 0, 12), (1, 1, 0, 12), (0, 1, 1, 12)]
    mse_dict_train = {}
    mse_dict_test = {}
    mse_member_dict_75 = {}
    mse_casual_dict_75 = {}
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
            train_mse, test_mse = mse.compare_mse_members(model, train_df,
                                                          test_df, 'member')
            pplot.prediction_plot_members(model, train_df, test_df, 'member',
                                          o_val[0], o_val[1], o_val[2], s_o[0],
                                          s_o[1], s_o[2], s_o[3])
            mse_member_dict_75[f'{o_val},{s_o}'] = {'Training MSE':
                                                "{:.2e}".format(train_mse),
                                                'Testing MSE':
                                                "{:.2e}".format(test_mse)}
            
            model = sm.tsa.statespace.SARIMAX(train_df['casual'],
                                              order=(o_val[0], o_val[1],
                                                     o_val[2]), 
                                              seasonal_order=(s_o[0], s_o[1],
                                                              s_o[2],
                                                              s_o[3])).fit()
            print(model.summary())
            train_mse, test_mse = mse.compare_mse_members(model, train_df,
                                                          test_df, 'casual')
            pplot.prediction_plot_members(model, train_df, test_df, 'casual',
                                          o_val[0], o_val[1], o_val[2], s_o[0],
                                          s_o[1], s_o[2], s_o[3])
            mse_member_dict_75[f'{o_val},{s_o}'] = {'Training MSE':
                                                "{:.2e}".format(train_mse),
                                                'Testing MSE':
                                                "{:.2e}".format(test_mse)}



def stationarity_autocorrelation_test_first_diff(data):
    """Tests stationarity and autocorrelarity of first difference"""
    print('The purpose of this test is to determine the stationarity and '
          'autocorrelarity\n of the first difference.')
    first_diff = f.order_difference(data)
    test_stationarity(first_diff['count'], window=12,
                      title="First Order Difference")
    pplot.acf_pacf(first_diff)


def stationarity_autocorrelation_test_second_diff(data):
    """Tests stationarity and autocorrelarity of second difference"""
    print('The purpose of this test is to determine the stationarity and '
          'autocorrelarity\n of the second difference.')
    first_diff = f.order_difference(data)
    second_diff = f.order_difference(first_diff)
    test_stationarity(second_diff['count'], window=12,
                      title="Second Order Difference")
    pplot.acf_pacf(second_diff)


def stationarity_test_seasonal_diff(data):
    """Tests stationarity and autocorrelarity of seasonal difference"""
    print('The purpose of this test is to determine the stationarity and '
          'autocorrelarity\n of the seasonal difference.')
    season = f.seasonal_difference(data)
    test_stationarity(season['count'], window=12,
                      title="Seasonal Difference")
    pplot.acf_pacf(season)


def stationarity_test_seasonal_first_diff(data):
    """Tests stationarity and autocorrelarity of seasonal difference"""
    print('The purpose of this test is to determine the stationarity and '
          'autocorrelarity\n of the seasonal difference of the first '
          'difference.')
    first_diff = f.order_difference(data)
    season_first = f.order_difference(first_diff)
    test_stationarity(season_first['count'], window=12,
                      title="Seasonal Difference of First Order Difference")
    pplot.acf_pacf(season_first)


def best_model_sarima(train_df, test_df):
    """Tests a range of models using sarima for best fit"""
    print('The purpose of this test is to determine the model with the best '
          'fit\n using Sarima.')
    mse_train, mse_test = sarima(train_df, test_df)


def forecast_original(data):
    """Creates forecast data for original data"""
    model_01112 = sm.tsa.statespace.SARIMAX(data['count'], trend='n', order=(2, 1, 2),
                                            seasonal_order=(0, 1, 1, 12)).fit()
    print('Model (2,1,2)(0,1,1,12):')
    pplot.forecast_plot(model_01112, data)
