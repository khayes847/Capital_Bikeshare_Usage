"""Set of general functions"""
import pandas as pd


def import_cleaned():
    """Imports data with correct index"""
    data = pd.read_csv('data/cleaned_for_testing.csv',
                       index_col='date_of_trip')
    data.index = pd.to_datetime(data.index, format='%Y/%m/%d')
    return data


def train_split(data, train_percent=.75):
    """Splits data into 75% training and test data"""
    train_index_m = int(len(data)*train_percent) + 1
    train_m = data.iloc[:train_index_m]
    test_m = data.iloc[train_index_m:]
    return train_m, test_m


def order_difference(data):
    """Creates data set with order difference"""
    data_diff = data.copy()
    data_diff['count'] = data['count'].diff()
    data_diff.dropna(inplace=True)
    return data_diff


def seasonal_difference(data):
    """Creates data set with seasonal difference"""
    season = data.copy()
    season['count'] = data['count'] - data['count'].shift(12)
    season.dropna(inplace=True)
    return season


def print_dict(dic):
    """Prints dictionary line by line"""
    for k_val, v_val in dic.items():
        print(f'{k_val}: p-value {v_val}')
