
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import gc
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

##############################################################################
# とりあえず↓の写経
# https://www.kaggle.com/ogrellier/i-have-seen-the-future/notebook
##############################################################################

warnings.simplefilter('error', SettingWithCopyWarning)
gc.enable()

# Define folding strategy
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids

def main(debug = False):
    num_rows = 10000 if debug else None

    # Get the extracted data
    train = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_train.gz',
                        dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
    test = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_test.gz',
                       dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)

    # Get session target
    y_reg = train['totals.transactionRevenue'].fillna(0)
    del train['totals.transactionRevenue']

    if 'totals.transactionRevenue' in test.columns:
        del test['totals.transactionRevenue']

    # Add date features
    train['target'] = y_reg
    for df in [train, test]:
        df['vis_date'] = pd.to_datetime(df['visitStartTime'], unit='s')
        df['sess_date_dow'] = df['vis_date'].dt.dayofweek
        df['sess_date_hours'] = df['vis_date'].dt.hour
        df['sess_date_dom'] = df['vis_date'].dt.day
        df.sort_values(['fullVisitorId', 'vis_date'], ascending=True, inplace=True)
        df['next_session_1'] = (
            df['vis_date'] - df[['fullVisitorId', 'vis_date']].groupby('fullVisitorId')['vis_date'].shift(1)
        ).astype(np.int64) // 1e9 // 60 // 60
        df['next_session_2'] = (
            df['vis_date'] - df[['fullVisitorId', 'vis_date']].groupby('fullVisitorId')['vis_date'].shift(-1)
        ).astype(np.int64) // 1e9 // 60 // 60

    #     df['max_visits'] = df['fullVisitorId'].map(
    #         df[['fullVisitorId', 'visitNumber']].groupby('fullVisitorId')['visitNumber'].max()
    #     )

        df['nb_pageviews'] = df['date'].map(
            df[['date', 'totals.pageviews']].groupby('date')['totals.pageviews'].sum()
        )

        df['ratio_pageviews'] = df['totals.pageviews'] / df['nb_pageviews']

    #     df['nb_sessions'] = df['date'].map(
    #         df[['date']].groupby('date').size()
    #     )

    #     df['nb_sessions_28_ma'] = df['date'].map(
    #         df[['date']].groupby('date').size().rolling(28, min_periods=7).mean()
    #     )

    #     df['nb_sessions_28_ma'] = df['nb_sessions'] / df['nb_sessions_28_ma']

    #     df['nb_sessions_per_day'] = df['date'].map(
    #         df[['date']].groupby('date').size()
    #     )

    #     df['nb_visitors_per_day'] = df['date'].map(
    #         df[['date','fullVisitorId']].groupby('date')['fullVisitorId'].nunique()
    #     )

    y_reg = train['target']
    del train['target']


if __name__ == '__main__':
    main()
