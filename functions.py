"""
This module provides general functions used in other modules and technical
notebook
"""
import pandas as pd


def import_cleaned():
    """
    Imports data with correct index.

    Parameters:
    None

    Returns:
    data: Imported dataset.
    """
    data = pd.read_csv('data/cleaned_for_testing.csv',
                       index_col='date_of_trip')
    data.index = pd.to_datetime(data.index, format='%Y/%m/%d')
    return data


def train_split(data, train_percent=.75):
    """
    Splits data into 75% training and 25% testing data.

    Parameters:
    data: Dataset in question.
    train_percent: Percentage of data that will be in training
    dataset.

    Returns:
    train_m: Training dataset.
    test_m: Test dataset.
    """
    train_index_m = int(len(data) * train_percent) + 1
    train_m = data.iloc[:train_index_m]
    test_m = data.iloc[train_index_m:]
    return train_m, test_m


def order_difference(data, var='count', diff=True, s_diff=False):
    """
    Takes time-series dataset, and returns the dataset with
    the specified number of regular and seasonal differences.

    Parameters:
    data: The time-series dataset in question.
    var: Variable to be differenced
    diff: Boolean, whether to add a one-month difference.
    s_diff: Boolean, whether to add a 12-month difference.

    Returns:
    data2: The time-series model with the
           specified differences enacted.
    """
    data2 = data.copy()
    if diff:
        data2[var] = data2[var].diff()
        data2.dropna(inplace=True)
        if s_diff:
            data2[var] = data2[var] - data2[var].shift(12)
            data2.dropna(inplace=True)
            return data2
        return data2
    data2[var] = data2[var] - data2[var].shift(12)
    data2.dropna(inplace=True)
    return data2


def master_breakdown():
    """
    Returns data divided between member vs. casual rentals.

    Parameters:
    None

    Returns:
    data: Imported dataset.
    """
    data = pd.read_csv('data/master_breakdown.csv',
                       index_col='date_of_trip')
    data = data.rename(columns={'Member type': 'member_type'})
    data['member_type'] = data['member_type'].map(lambda x: x.lower())
    data = data.groupby(['date_of_trip', 'member_type']).min()
    data = pd.pivot_table(data, values='count', columns='member_type',
                          index='date_of_trip').drop(columns=['unknown'])
    data.index = pd.to_datetime(data.index, format='%Y/%m/%d')
    data = data.resample('m').sum()
    return data
