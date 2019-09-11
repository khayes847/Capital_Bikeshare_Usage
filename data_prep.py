#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:41:38 2019

@author: marissaeppes
"""
import pandas as pd


def manage_cols(data):
    """Groups rows by date"""
    print('starting manage_cols')
    data['Start date'] = pd.to_datetime(data['Start date'])
    data['date_of_trip'] = [item.date() for item in data['Start date']]
    data = data.groupby('date_of_trip').count()
    return data


def data_by_date(year):
    """Merges seperate data sets"""
    if year == 2018:
        df_concat = pd.DataFrame()
        for month in ['01', '02', '03', '04', '05', '06', '07',
                      '08', '09', '10', '11', '12']:
            if month == '11':
                path = (f'data/2018-capitalbikeshare-tripdata/2018{month}'
                        '-capitalbikeshare-tripdata.csv')
                print(path)
                data = pd.read_csv(path)
                data = data.iloc[0:202376]
                data = manage_cols(data)
                df_concat = pd.concat([df_concat, data])
            else:
                path = (f'data/2018-capitalbikeshare-tripdata/2018{month}'
                        '-capitalbikeshare-tripdata.csv')
                print(path)
                data = pd.read_csv(path)
                data = manage_cols(data)
                df_concat = pd.concat([df_concat, data])
    elif year == 2019:
        df_concat = pd.DataFrame()
        for month in ['01', '02', '03', '04', '05', '06', '07']:
            path = (f'data/2019-capitalbikeshare-tripdata/2019{month}'
                    '-capitalbikeshare-tripdata.csv')
            print(path)
            data = pd.read_csv(path)
            data = manage_cols(data)
            df_concat = pd.concat([df_concat, data])
    elif year in [2010, 2011]:
        path = f'data/{year}-capitalbikeshare-tripdata.csv'
        print(path)
        data = pd.read_csv(path)
        data = manage_cols(data)
        df_concat = data
    else:
        df_concat = pd.DataFrame()
        for quart in ['Q1', 'Q2', 'Q3', 'Q4']:
            path = (f'data/{year}-capitalbikeshare-tripdata/{year}{quart}'
                    '-capitalbikeshare-tripdata.csv')
            print(path)
            data = pd.read_csv(path)
            data = manage_cols(data)
            df_concat = pd.concat([df_concat, data])
    return df_concat


def merge_data():
    """Merges all bike datasets"""
    master = pd.DataFrame()
    years = list(range(2010, 2020))
    for year in years:
        data = data_by_date(year)
        master = pd.concat([master, data])
    return master


def drop_columns(data):
    """Drops unnecessary columns"""
    print('drop_columns')
    data.drop(columns=['Start date', 'End date', 'Start station number',
                       'Start station', 'End station number', 'End station',
                       'Bike number', 'Member type'], inplace=True)
    return data


def rename_columns(data):
    """Renames data columns"""
    print('rename_columns')
    data = data.rename(columns={'Duration': 'count'})
    return data


def date_of_trip_changes(data):
    """Changes date_of_trip to datetime, resets as index,
    resamples as monthly"""
    print('date_of_trip_changes')
    data = data.reset_index()
    data.date_of_trip = pd.to_datetime(data.date_of_trip, format='%Y/%m/%d')
    data = data.set_index('date_of_trip')
    data = data.resample('m').sum()
    return data


def add_data(data, master):
    """Adds weather data set to existing data set as column"""
    print(f'add_data_{data}')
    new_df = pd.read_csv(f'data/{data}.csv')
    new_df = new_df.rename(columns={'Unnamed: 0': 'date'})
    new_df = new_df.set_index('date')
    new_df = new_df.stack(level=-1, dropna=True)
    master[str(data)] = list(new_df)
    return master


def full_clean():
    """Runs all the prior functions and exports csv"""
    cleaning_data_1 = merge_data()
    cleaning_data_1.to_csv('./data/cleaning_data_1.csv', index=True)
    cleaning_data_2 = rename_columns(cleaning_data_1)
    cleaning_data_3 = drop_columns(cleaning_data_2)
    cleaning_data_4 = rename_columns(cleaning_data_3)
    cleaning_data_5 = date_of_trip_changes(cleaning_data_4)
    cleaning_data_6 = add_data('dc_temps', cleaning_data_5)
    cleaned_data = add_data('precipitation', cleaning_data_6)
    cleaned_data.to_csv('./data/cleaned_for_testing.csv', index=True)
    return cleaned_data
