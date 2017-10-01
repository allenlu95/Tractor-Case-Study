import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import timedelta
import numpy as np
'''
All methods described in EDA notebook
'''
def funnel_convert(df):
    df = df.loc[:,df.count()>300000]
    df = df.loc[:,[(len(df[column].unique())<10) if df[column].dtype.name == 'object' else True for column in df.columns]]
    df = df[['SalesID', 'SalePrice', 'MachineID', 'ModelID', 'datasource','YearMade', 'ProductGroup','Enclosure', 'year_change', 'saledate_converted', 'equipment_age']]
    return df
def change_dates(df):
    df['year_change'] = df['YearMade'] == 1000
    mean = df[df['YearMade']!=1000]['YearMade'].mean()
    df['YearMade'] = df['YearMade'].replace(1000, mean)
    df['saledate_converted'] = pd.to_datetime(df.saledate)
    return df
def fill_na(df):
    df['Enclosure'] = df['Enclosure'].fillna(method='ffill')
    return df
def ComputeAge(df):
    saledate = pd.to_datetime(df.saledate_converted)
    df['equipment_age'] = saledate.dt.year-df.YearMade
    return df
def final_features(df):
    df = df[['SalePrice', 'YearMade', 'ProductGroup', 'Enclosure', 'year_change','equipment_age']]
    df = pd.get_dummies(df, drop_first = True)
    return df

df = (pd.read_csv('data/train.csv')
      .pipe(change_dates)
      .pipe(fill_na)
      .pipe(ComputeAge)
      .pipe(funnel_convert)
      )

outfile = 'data/clean.csv'
df.to_csv(outfile, index=False)
