"""
This module provides all plotting functions for both parts of the project:
1) Time series analysis on all monthly rentals
2) Time series analysis on broken down data between monthly member rentals and
   monthly casual rentals
"""
# pylint: disable=unused-variable
# pylint: disable=invalid-name

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import functions as f


def initial_plot(data):
    """
    Plots initial time series data prior to any analysis.

    Parameters:
    data: Dataset in question.

    Returns:
    None
    """
    fig, ax_val = plt.subplots(figsize=(18, 8))

    ax_val.plot(data.index.values, data['count'] / 100000,
                lw=2, color='mediumseagreen')

    ax_val.set_xlabel('Year', fontsize=20)
    ax_val.set_ylabel('Ride Count (100k)', fontsize=20)

    ax_val.xaxis.set_major_locator(mdates.YearLocator())
    ax_val.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.title('Number of Bike Rentals per Month, Time Series', fontsize=30)
    plt.show()


def split_plot(train, test):
    """
    Plots both train and test data on the same graph.

    Parameters:
    train: Training dataset.
    test: Test dataset.

    Returns:
    None
    """
    fig = plt.figure(figsize=(15, 8))

    plt.plot(train['count'] / 100000, lw=3)
    plt.plot(test['count'] / 100000, lw=3, color='navy')
    plt.legend(['Train', 'Test'], prop={'size': 24})

    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Ride Count (100k)', fontsize=20)

    plt.title('Number of Bike Rentals per Month, Time Series', fontsize=30)
    plt.show()


def decomposition(data):
    """
    Graphs seasonal decomposition.

    Parameters:
    data: Dataset being used.

    Returns:
    None
    """
    decomposition_data = seasonal_decompose(data['count'] / 100000, freq=12)
    fig = plt.figure()
    fig = decomposition_data.plot()
    fig.set_size_inches(15, 8)


def stationarity_plot(timeseries, window, title):
    """
    Calculates and plots rolling statistics.

    Parameters:
    timeseries: SARIMA model.
    window: Window of months used for plot.
    title: Title of graph.

    Returns:
    None
    """
    # Determining rolling statistics
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries.iloc[window:] / 100000,
                    color='blue', label='Original')
    mean = plt.plot(rolmean / 100000, color='red', label='Rolling Mean')
    std = plt.plot(rolstd / 100000, color='black', label='Rolling Std')
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Ride Count', fontsize=20)
    plt.legend(loc='best')
    plt.title(title)
    plt.show()


def acf_pacf(data, var='count'):
    """
    Plots autocorrelation function (ACF) and partial autocorrelation function
    (PACF) outputs.

    Parameters:
    data: Dataset being used.
    var: Dependent variable.

    Returns:
    None
    """
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data[var].iloc[1:], lags=13, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data[var].iloc[1:], lags=13, ax=ax2)


def prediction_plot(model, data, orders, test=False, var='count'):
    """
    Plots model prediction on top of original data for both training and
    testing data for a given model. This is to act as a visual aid and not an
    indicator of which model is most optimal. This function is for part 1 of
    the project (all monthly rentals).

    Parameters:
    model: SARIMA model being used.
    data: Dataset being used.
    orders: Number of regular and seasonal orders.
    test: Whether we will also plot the test dataset.
    var: Dependent variable.

    Returns:
    None
    """
    if test:
        data, testing_set = f.train_split(data)
        predict_test = model.forecast(len(testing_set))

    predict_train = model.predict()

    fig = plt.figure(figsize=(20, 20))
    ax1 = plt.subplot(211)
    if test:
        ax2 = plt.subplot(212)

    ax1.plot(data.index, data[var],
             lw=4, color='mediumturquoise')
    if test:
        ax1.plot(testing_set.index, testing_set[var], lw=4, color='blue')
    ax1.plot(data.index, predict_train, lw=4, color='magenta')

    if test:
        ax2.plot(data.index, data[var],
                 lw=4, color='mediumturquoise')
        ax2.plot(testing_set.index, testing_set[var], lw=4, color='blue')
        ax2.plot(testing_set.index, predict_test, lw=4, color='orange')

    ax1.set_xlabel('Date', fontsize=20)
    ax1.set_ylabel('Count', fontsize=20)

    if test:
        ax2.set_xlabel('Date', fontsize=20)
        ax2.set_ylabel('Count', fontsize=20)

    ax1.set_title(f'Number of Bike Rentals per Month Time Series, '
                  f'SARIMA {orders[0]}{orders[1]}', fontsize=30)
    if test:
        ax2.set_title(f'Number of Bike Rentals per Month Time Series, '
                      f'SARIMA {orders[0]}{orders[1]}', fontsize=30)

    ax1.legend(['Train',
                'Model Prediction on Training Set'], prop={'size': 24})
    if test:
        ax2.legend(['Train', 'Test',
                    'Model Prediction on Testing Set'], prop={'size': 24})
    plt.show()


def forecast_plot(model, master, n_forecast=12, predict_int_alpha=.2,
                  var='count'):
    """
    Plots predicted forecast and prediction interval alongside original data.
    This function is for part 1 of the project (all monthly rentals).

    Parameters:
    model: SARIMA model being used.
    master: Dataset being used.
    n_forecast: Number of months being forecasted.
    predict_int_alpha: Alpha for confidence interval
    var: Dependent variable.

    Returns:
    None
    """
    fig = plt.figure(figsize=(18, 9))
    plt.plot(master.index, master[var] / 100000,
             lw=4, color='mediumorchid')
    forecast = model.forecast(n_forecast)
    predict_int = model.get_forecast(n_forecast
                                     ).conf_int(alpha=predict_int_alpha)
    forecast_dates = pd.date_range(master.index[-1], freq='m',
                                   periods=n_forecast + 1).tolist()[1:]
    plt.plot(forecast_dates, forecast / 100000, lw=6, color='indigo')
    plt.plot(forecast.index, predict_int[f'lower {var}'] / 100000,
             color='darkgray', lw=4)
    plt.plot(forecast.index, predict_int[f'upper {var}'] / 100000,
             color='darkgray', lw=4)
    plt.fill_between(forecast.index, predict_int[f'lower {var}'] / 100000,
                     predict_int[f'upper {var}'] / 100000, color='darkgray')
    plt.xlim([master.index[0], forecast.index[-1]])
    plt.ylim(0)
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Ride Count (100k)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(['Data', 'Forecast', '80% Prediction Interval'],
               prop={'size': 24})
    plt.title(f'{n_forecast}-Month Forecast on Bike Rentals per Month',
              fontsize=30)
    plt.show()


def initial_breakdown_graph(data):
    """
    Plots initial time series data broken down by member and casual rentals on
    the same graph prior to any analysis (Part 2).

    Parameters:
    data: Dataset being used.

    Returns:
    None
    """
    fig = plt.figure(figsize=(18, 8))
    plt.plot(data.casual / 100000, lw=5, color='mediumorchid')
    plt.plot(data.member / 100000, lw=5, color='green')
    plt.xlabel('Year', fontsize=25)
    plt.ylabel('Ride Count (100k)', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(['Casual', 'Member'], prop={'size': 24})
    plt.xlim([data.index[0], data.index[-1]])
    plt.ylim(0)
    plt.title('Number of Casual and Member Rides per Month', fontsize=30)
    plt.show()


def prediction_plot_breakdown(model, training_set, testing_set, kind,
                              o_l):
    """
    Plots model prediction on top of original data for both training and
    testing data for a given model. This is to act as a visual aid and not an
    indicator of which model is most optimal. This function is for part 2 of
    the project (broken down monthly rentals between member and casual).

    Parameters:
    model: SARIMA model being used.
    training_set: Training dataset.
    testing_set: Test dataset.
    kind: Dependent variable.
    o_l: ARIMA and SARIMA models.

    Returns:
    None
    """

    if o_l[1][3] == 52:
        freq = 'Week'
        lw = 2
    elif o_l[1][3] == 12:
        freq = 'Month'
        lw = 4
    else:
        lw = 2
        freq = '(test)'

    order = f'({o_l[0][0]},{o_l[0][1]},{o_l[0][2]})'
    seasonal_order = f'({o_l[1][0]},{o_l[1][1]},{o_l[1][2]},{o_l[1][3]})'

    predict_train = model.predict()
    predict_test = model.forecast(len(testing_set))

    fig = plt.figure(figsize=(20, 20))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    ax1.plot(training_set.index,
             training_set[kind], lw=lw, color='mediumturquoise')
    ax1.plot(testing_set.index, testing_set[kind], lw=lw, color='blue')
    ax1.plot(training_set.index, predict_train, lw=lw, color='magenta')

    ax2.plot(training_set.index,
             training_set[kind], lw=lw, color='mediumturquoise')
    ax2.plot(testing_set.index, testing_set[kind], lw=lw, color='blue')
    ax2.plot(testing_set.index, predict_test, lw=lw, color='orange')

    ax1.set_xlabel('Date', fontsize=20)
    ax1.set_ylabel('Count', fontsize=20)

    ax2.set_xlabel('Date', fontsize=20)
    ax2.set_ylabel('Count', fontsize=20)

    ax1.set_title(
        f'Number of {kind} Bike Rentals per {freq} Time Series, SARIMA '
        f'{order}{seasonal_order}', fontsize=30)
    ax2.set_title(
        f'Number of {kind} Bike Rentals per {freq} Time Series, SARIMA'
        f' {order}{seasonal_order}',
        fontsize=30)

    ax1.legend(['Train', 'Test', 'Model Prediction on Training Set'],
               prop={'size': 24})
    ax2.legend(['Train', 'Test', 'Model Prediction on Testing Set'],
               prop={'size': 24})
    plt.show()


def model_plot_breakdown(model, master, kind):
    """
    Plots initial data alongside prediction (dotted) for any given model. This
    is to act as a visual aid and not an indicator of which model is most
    optimal. This function is for part 2 of the project (broken down monthly
    rentals between member and casual). This differs from
    prediction_plot_breakdown in that original data is not split between train
    and test and only one graph is output for this function.

    Parameters:
    model: SARIMA model being used.
    master: Dataset being used.
    kind: Dependent variable.

    Returns:
    None
    """
    fig = plt.figure(figsize=(18, 9))
    plt.plot(master[kind.lower()] / 100000, lw=4, color='royalblue')
    plt.plot(master.index, model.predict() / 100000, lw=5, color='black',
             ls=':')
    plt.title(
        f'Number of {kind} Bike Rentals per Month: Model of Historical Data',
        fontsize=30)
    plt.xlabel('Year', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Ride Count (100k)', fontsize=25)
    plt.xlim([master.index[0], master.index[-1]])
    plt.ylim(0)
    plt.legend(['Data', 'Model'], prop={'size': 24})
    plt.show()


def model_plot_breakdown_both(model_member, model_casual, master):
    """
    Plots initial data alongside prediction (dotted) for any given model and
    plots member and casual data/predictions on the same graph. This is to act
    as a visual aid and not an indicator of which model is most optimal. This
    function is for part 2 of the project (broken down monthly rentals between
    member and casual).

    Parameters:
    model_member: SARIMA model for 'Member' dataset.
    model_casual: SARIMA model for 'Casual' dataset.
    master: Overall dataset.

    Returns:
    None
    """
    fig = plt.figure(figsize=(18, 9))
    plt.plot(master.member / 100000, lw=4, color='mediumseagreen')
    plt.plot(master.casual / 100000, lw=4, color='mediumorchid')
    plt.plot(master.index, model_member.predict() /
             100000, lw=5, color='black', ls=':')
    plt.plot(master.index, model_casual.predict() /
             100000, lw=5, color='black', ls=':')
    plt.title(
        f'Number of Member and Casual Bike Rentals per Month: Model of '
        f'Historical Data',
        fontsize=30)
    plt.xlabel('Year', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('Ride Count (100k)', fontsize=25)
    plt.xlim([master.index[0], master.index[-1]])
    plt.ylim(0)
    plt.legend(['Member Data', 'Casual Data', 'Models'], prop={'size': 24})
    plt.show()


def forecast_plot_breakdown_both(model_member, model_casual, master,
                                 n_forecast, predict_int_alpha=.2):
    """
    Plots predicted forecast and prediction interval alongside original data.
    Plots both the member forecasts and casual forecasts on the same graph.
    This function is for part 2 of the project (broken down monthly rentals
    between member and casual).

    Parameters:
    model_member: SARIMA model for 'Member' dataset.
    model_casual: SARIMA model for 'Casual' dataset.
    master: Overall dataset.
    n_forecast: Number of months to forecast.
    predict_int_alpha: Alpha for forecast confidence interval.

    Returns:
    None
    """
    fig = plt.figure(figsize=(18, 9))
    plt.plot(master.index, master['member'] /
             100000, lw=4, color='mediumseagreen')
    plt.plot(master.index, master['casual'] / 100000, lw=4,
             color='mediumorchid')

    forecast_member = model_member.forecast(n_forecast)
    forecast_casual = model_casual.forecast(n_forecast)
    predict_int_member = model_member.get_forecast(
        n_forecast).conf_int(alpha=predict_int_alpha)
    predict_int_casual = model_casual.get_forecast(
        n_forecast).conf_int(alpha=predict_int_alpha)
    forecast_dates = pd.date_range(
        master.index[-1], freq='m', periods=n_forecast + 1).tolist()[1:]

    plt.plot(forecast_dates, forecast_member / 100000, lw=6, color='darkgreen')
    plt.plot(forecast_dates, forecast_casual / 100000, lw=6, color='indigo')

    plt.plot(forecast_casual.index,
             predict_int_casual['lower casual'] / 100000, color='thistle',
             lw=4)
    plt.plot(forecast_casual.index,
             predict_int_casual['upper casual'] / 100000, color='thistle',
             lw=4)

    plt.plot(forecast_member.index,
             predict_int_member['lower member'] / 100000, color='powderblue',
             lw=4)
    plt.plot(forecast_member.index,
             predict_int_member['upper member'] / 100000, color='powderblue',
             lw=4)

    plt.fill_between(forecast_casual.index,
                     predict_int_casual['lower casual'] /
                     100000, predict_int_casual['upper casual'] / 100000,
                     color='thistle')
    plt.fill_between(forecast_member.index,
                     predict_int_member['lower member'] /
                     100000, predict_int_member['upper member'] / 100000,
                     color='powderblue')

    plt.xlim([master.index[0], forecast_member.index[-1]])
    plt.ylim(0)
    plt.xlabel('Year', fontsize=25)
    plt.ylabel('Ride Count (100k)', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(['Member Rides', 'Casual Rides', 'Forecast on Member Rides',
                'Forecast on Casual Rides'], prop={'size': 24})
    plt.title(
        f'{n_forecast}-Month Forecast on Bike Rentals per Month', fontsize=30)
