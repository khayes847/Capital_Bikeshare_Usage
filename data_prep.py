"""
Module gathers and prepares data for analysis for both parts of project:

1) Time series analysis on all monthly rentals
2) Time series analysis on broken down data between monthly member rentals and
   monthly casual rentals
"""

import pandas as pd


def manage_cols(data):
    """
    Groups rows by date and provides a daily count of rides.

    Parameters:
    data: Dataset in question.

    Returns:
    data: Dataset with grouped rows.
    """
    print('starting manage_cols')
    data['Start date'] = pd.to_datetime(data['Start date'])
    data['date_of_trip'] = [item.date() for item in data['Start date']]
    data = data.groupby('date_of_trip').count()
    return data


def data_by_date(year):
    """
    Collects datasets for each year. Some years are organized differently from
    others and therefore require customized paths.

    Parameters:
    year: String value for year in question.

    Returns:
    data: Database for year.
    """
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
    """
    Concatenates each yearly dataframe into a single master dataframe.

    Parameters:
    None

    Returns:
    master: Master dataset.
    """
    master = pd.DataFrame()
    years = list(range(2010, 2020))
    for year in years:
        data = data_by_date(year)
        master = pd.concat([master, data])
    return master


def drop_columns(data):
    """
    Drops unnecessary columns.

    Parameters:
    data: Dataset in question.

    Returns:
    data: Dataset with dropped columns.
    """
    print('drop_columns')
    data.drop(columns=['Start date', 'End date', 'Start station number',
                       'Start station', 'End station number', 'End station',
                       'Bike number', 'Member type'], inplace=True)
    return data


def rename_columns(data):
    """
    Renames the number of rentals attribute to 'count'. Because a
    groupby('date_of_trip').count() was performed on the original data, the
    'Duration' column, which was the only column left after the drop, actually
    corresponds to the number of rides in a given day.

    Parameters:
    data: Dataset in question.

    Returns:
    data: Dataset with renamed columns.
    """
    print('rename_columns')
    data = data.rename(columns={'Duration': 'count'})
    return data


def date_of_trip_changes(data):
    """
    Changes date_of_trip to datetime object, resets datetime as index,
    resamples data on monthly basis.

    Parameters:
    data: Dataset in question.

    Returns:
    data: Dataset with resampled data on monthly basis.
    """
    print('date_of_trip_changes')
    data = data.reset_index()
    data.date_of_trip = pd.to_datetime(data.date_of_trip, format='%Y/%m/%d')
    data = data.set_index('date_of_trip')
    data = data.resample('m').sum()
    return data


def full_clean():
    """
    Runs all the prior functions and saves/returns cleaned master dataset for
    original analysis.

    Parameters:
    None

    Returns:
    cleaned_data: Fully-cleaned dataset.
    """
    cleaning_data_1 = merge_data()
    cleaning_data_2 = drop_columns(cleaning_data_1)
    cleaning_data_3 = rename_columns(cleaning_data_2)
    cleaned_data = date_of_trip_changes(cleaning_data_3)
    cleaned_data.to_csv('./data/cleaned_for_testing.csv', index=True)
    return cleaned_data


def manage_cols_breakdown(data):
    """
    Groups rows by date and drops all columns except 'Member type'.

    Parameters:
    data: Dataset in question.

    Returns:
    data: Altered dataset with rows grouped by date.
    """
    data['Start date'] = pd.to_datetime(data['Start date'])
    data['date_of_trip'] = [item.date() for item in data['Start date']]
    data.drop(columns=['Start date', 'End date', 'Start station number',
                       'Start station', 'End station number', 'End station',
                       'Bike number'], inplace=True)
    data = data.rename(columns={'Duration': 'count'})
    return data


def data_by_date_breakdown(year):
    """
    Creates dataframe for given year to be used in determining members vs.
    casual rentals. Some years are organized differently from others and
    therefore require customized paths.

    Parameters:
    year: Dataset in question.

    Returns:
    data_concat: Created dataset.
    """
    if year == '2018':
        data_concat = pd.DataFrame()
        for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09',
                      '10', '11',
                      '12']:
            if month == '11':
                path = 'data/2018-capitalbikeshare-tripdata/' + year + \
                       month + '-capitalbikeshare-tripdata.csv'
                data = pd.read_csv(path)
                data = data.iloc[0:202376]
                data = manage_cols_breakdown(data)
                data = data.groupby(['date_of_trip', 'Member type']).count()
                data_concat = pd.concat([data_concat, data])
            else:

                path = 'data/2018-capitalbikeshare-tripdata/' + year + \
                       month + '-capitalbikeshare-tripdata.csv'
                data = pd.read_csv(path)
                data = manage_cols_breakdown(data)
                data = data.groupby(['date_of_trip', 'Member type']).count()
                data_concat = pd.concat([data_concat, data])
    elif year == '2019':
        data_concat = pd.DataFrame()
        for month in ['01', '02', '03', '04', '05', '06', '07']:
            path = 'data/2019-capitalbikeshare-tripdata/' + year + month + \
                   '-capitalbikeshare-tripdata.csv'
            data = pd.read_csv(path)
            data = manage_cols_breakdown(data)
            data = data.groupby(['date_of_trip', 'Member type']).count()
            data_concat = pd.concat([data_concat, data])
    elif year in ['2010', '2011']:
        path = f'data/{year}-capitalbikeshare-tripdata.csv'
        data = pd.read_csv(path)
        data = manage_cols_breakdown(data)
        data = data.groupby(['date_of_trip', 'Member type']).count()
        data_concat = data
    else:
        data_concat = pd.DataFrame()
        for q_val in ['Q1', 'Q2', 'Q3', 'Q4']:
            path = f'data/{year}-capitalbikeshare-tripdata/' \
                   f'{year}{q_val}-capitalbikeshare-tripdata.csv'
            data = pd.read_csv(path)
            data = manage_cols_breakdown(data)
            data = data.groupby(['date_of_trip', 'Member type']).count()
            data_concat = pd.concat([data_concat, data])
    return data_concat


def save_master_breakdown_df():
    """
    Merges dataframes across all years for member vs. casual rentals. Saves
    master dataframe as csv.

    Parameters:
    None

    Returns:
    None
    """
    data_dict = {}
    for year in ['2010', '2011', '2012', '2013', '2014', '2015', '2016',
                 '2017', '2018', '2019']:
        data_dict[year] = data_by_date_breakdown(year)
    master_breakdown = pd.DataFrame()
    for year in data_dict:
        master_breakdown = pd.concat([master_breakdown, data_dict[year]])
    master_breakdown.to_csv('data/master_breakdown.csv')
