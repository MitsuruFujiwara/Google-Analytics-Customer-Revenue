
import json
import pandas as pd
import numpy as np
import time
import pickle
import logging

from pandas.io.json import json_normalize
from multiprocessing import Pool as Pool
from contextlib import contextmanager

"""
Utility的なものを置いとくところ
"""

KEYS_FOR_FIELD = {'device': [
                        'browser',
                        'browserSize',
                        'browserVersion',
                        'deviceCategory',
                        'flashVersion',
                        'isMobile',
                        'language',
                        'mobileDeviceBranding',
                        'mobileDeviceInfo',
                        'mobileDeviceMarketingName',
                        'mobileDeviceModel',
                        'mobileInputSelector',
                        'operatingSystem',
                        'operatingSystemVersion',
                        'screenColors',
                        'screenResolution'
                        ],
                  'geoNetwork': [
                        'city',
                        'cityId',
                        'continent',
                        'country',
                        'latitude',
                        'longitude',
                        'metro',
                        'networkDomain',
                        'networkLocation',
                        'region',
                        'subContinent'
                        ],
                  'totals': [
                        'bounces',
                        'hits',
                        'newVisits',
                        'pageviews',
                        'transactionRevenue',
                        'visits'
                        ],
                  'trafficSource': [
                        'adContent',
                        'adwordsClickInfo',
                        'campaign',
                        'campaignCode',
                        'isTrueDirect',
                        'keyword',
                        'medium',
                        'referralPath',
                        'source'
                        ],
                        }

EXCLUDED_FEATURES = ["visitNumber", "date", "fullVisitorId", "sessionId", "visitId",
                     "visitStartTime", 'index', 'IS_TEST']
                     
def apply_func_on_series(data=None, func=None):
    return data.apply(lambda x: func(x))

def multi_apply_func_on_series(df=None, func=None, n_jobs=4):
    p = Pool(n_jobs)
    f_ = p.map(functools.partial(apply_func_on_series, func=func),
               np.array_split(df, n_jobs))
    f_ = pd.concat(f_, axis=0, ignore_index=True)
    p.close()
    p.join()
    return f_.values

def convert_to_dict(x):
    return eval(x.replace('false', 'False')
                .replace('true', 'True')
                .replace('null', 'np.nan'))

def get_dict_field(x_, key_):
    try:
        return x_[key_]
    except KeyError:
        return np.nan

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# correlation高い変数を削除する機能
def removeCorrelatedVariables(data, threshold):
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    col_drop = [column for column in upper.columns if any(upper[column] > threshold) & ('TARGET' not in column)]
    return col_drop

# 欠損値の率が高い変数を削除する機能
def removeMissingVariables(data, threshold):
    missing = (data.isnull().sum() / len(data)).sort_values(ascending = False)
    col_missing = missing.index[missing > 0.75]
    col_missing = [column for column in col_missing if 'TARGET' not in column]
    return col_missing

def save2pkl(path, df):
    f = open(path, 'wb')
    pickle.dump(df, f)
    f.close

def loadpkl(path):
    f = open(path, 'rb')
    out = pickle.load(f)
    return out

# 拾い物 https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields/notebook
def load_df(csv_path, nrows=None):

    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path,
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [column+'.'+subcolumn for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df

def line_notify(message):
    f = open('../input/line_token.txt')
    token = f.read()
    f.close
    line_notify_token = token.replace('\n', '')
    line_notify_api = 'https://notify-api.line.me/api/notify'

    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}  # 発行したトークン
    line_notify = requests.post(line_notify_api, data=payload, headers=headers)
