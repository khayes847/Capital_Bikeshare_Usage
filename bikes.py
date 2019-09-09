#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:41:38 2019

@author: marissaeppes
"""
import pandas as pd




def manage_cols(df):
    df['Start date'] = pd.to_datetime(df['Start date'])
    df['date_of_trip'] = [item.date() for item in df['Start date']]
    df.drop(columns = ['Start date', 'End date', 'Start station number', 
                       'Start station','End station number', 'End station',
                       'Bike number', 'Member type'], inplace = True)
    df = df.rename(columns = {'Duration':'count'})
    return df


def data_by_date(year):
    if year == '2018':
        df_concat = pd.DataFrame()
        for month in ['01','02','03','04','05','06','07','08','09','10','11',
                      '12']:
            if month == '11':
                path = year + month + '-capitalbikeshare-tripdata.csv'
                df = pd.read_csv(path)
                df = df.iloc[0:202376]
                df = manage_cols(df)
                df = df.groupby('date_of_trip').count()
                df_concat = pd.concat([df_concat, df])
            else:
                
                path = year + month + '-capitalbikeshare-tripdata.csv'
                df =pd.read_csv(path)
                df = manage_cols(df)
                df = df.groupby('date_of_trip').count()
                df_concat = pd.concat([df_concat, df])
            
    elif year == '2019':
        df_concat = pd.DataFrame()
        for month in ['01','02','03','04','05','06','07']:
            path = year + month + '-capitalbikeshare-tripdata.csv'
            df =pd.read_csv(path)
            df = manage_cols(df)
            df = df.groupby('date_of_trip').count()
            df_concat = pd.concat([df_concat, df])
    elif year == '2010' or year == '2011':
        path = f'{year}-capitalbikeshare-tripdata.csv'
        df =pd.read_csv(path)
        df = manage_cols(df)
        df = df.groupby('date_of_trip').count()
        df_concat = df

        
    else:
        df_concat = pd.DataFrame()
        for q in ['Q1','Q2','Q3','Q4']:
            path = f'{year}-capitalbikeshare-tripdata/{year}{q}-capitalbikeshare-tripdata.csv'
            df =pd.read_csv(path)
            df = manage_cols(df)
            df = df.groupby('date_of_trip').count()
            df_concat = pd.concat([df_concat, df])
    return df_concat

def concat_data(year, df_to_concat=None):
    if df_to_concat == None:
        df_to_concat = pd.DataFrame()
    else:
        pass
    if year == '2018':
        for month in ['01','02','03','04','05','06','07','08','09','10','11',
                      '12']:
            path = year + month + '-capitalbikeshare-tripdata.csv'
            df =pd.read_csv(path)
            master = pd.concat([df_to_concat, df])
    elif year == '2019':
        for month in ['01','02','03','04','05','06','07']:
            path = year + month + '-capitalbikeshare-tripdata.csv'
            df =pd.read_csv(path)
            master = pd.concat([df_to_concat, df])
    else:
        for q in ['Q1','Q2','Q3','Q4']:
            path = f'{year}-capitalbikeshar-tripdata\{year}{q}-capitalbikeshare-tripdata.csv'
            df =pd.read_csv(path)
            master = pd.concat([df_to_concat, df])
    return master

