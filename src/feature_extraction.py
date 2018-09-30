
import json
import pandas as pd
import numpy as np

from multiprocessing import Pool
from pandas.io.json import json_normalize
from contextlib import contextmanager
from Utils import save2pkl, loadpkl
import os

"""
特徴量抽出用のスクリプト
"""

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

if __name__ == '__main__':
    if os.path.isfile('../output/train_df.pkl'):
        train_df = loadpkl('../output/train_df.pkl')
        test_df = loadpkl('../output/test_df.pkl')
    else:
        train_df = load_df(csv_path='../input/train.csv')
        test_df = load_df(csv_path='../input/test.csv')

        save2pkl('../output/train_df.pkl', train_df)
        save2pkl('../output/test_df.pkl', test_df)

    df = train_df.append(test_df)
    print(df.head(10))
