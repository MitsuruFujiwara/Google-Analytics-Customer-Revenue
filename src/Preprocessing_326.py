import pandas as pd
import gc
import numpy as np

from sklearn.preprocessing import LabelEncoder
from Utils import load_df, EXCLUDED_FEATURES, one_hot_encoder

def get_df(num_rows=None):
    # load datasets
    train_df = load_df('../input/train.csv', nrows=num_rows)
    test_df = load_df('../input/test.csv', nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))

    # train testの識別用
    train_df['IS_TEST'] = False
    test_df['IS_TEST'] = True

    # Merge
    df = train_df.append(test_df).reset_index()

    del train_df, test_df
    gc.collect()

    # 不要カラムを抽出
    for col in df.columns:
        if len(df[col].value_counts()) == 1:
            EXCLUDED_FEATURES.append(col)

    # numeric columnsの抽出
    num_cols = [c for c in df.columns if c.startswith("total") and c not in EXCLUDED_FEATURES]

    # numeric columnsを数値型へ変換
    df[num_cols] = df[num_cols].astype(float)

    # fillna
    df[num_cols] = df[num_cols].fillna(0)

    # categorical columnsの処理
    df, cat_cols = one_hot_encoder(df, nan_as_category=True)

    """
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
    """
#     df['max_visits'] = df['fullVisitorId'].map(
#         df[['fullVisitorId', 'visitNumber']].groupby('fullVisitorId')['visitNumber'].max()
#     )
    """
    df.loc[:,'totals.pageviews'] = df.loc[:,'totals.pageviews'].astype('float64')
    df['nb_pageviews'] = df['date'].map(
        df[['date', 'totals.pageviews']].groupby('date')['totals.pageviews'].sum()
    )

    df['ratio_pageviews'] = df['totals.pageviews'] / df['nb_pageviews']
    """

    df = df[cat_cols+num_cols+['IS_TEST']]
    return df

if __name__ == '__main__':
    # test
    df = get_df(num_rows=10000)
    print(df)
