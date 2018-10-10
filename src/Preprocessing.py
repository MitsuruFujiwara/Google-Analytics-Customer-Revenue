import pandas as pd
import gc
import numpy as np

from sklearn.preprocessing import LabelEncoder
from Utils import load_df, EXCLUDED_FEATURES

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

    # TODO: ここから下にFeature Engineeringの処理追加していく感じで

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

    df.loc[:,'totals.pageviews'] = df.loc[:,'totals.pageviews'].astype('float64')
    df['nb_pageviews'] = df['date'].map(
        df[['date', 'totals.pageviews']].groupby('date')['totals.pageviews'].sum()
    )
    
    df['ratio_pageviews'] = df['totals.pageviews'] / df['nb_pageviews']

    # とりあえず最低限必要な処理のみ

    # 使用しないカラムを定義
    constant_columns = []
    for col in df.columns:
        if len(df[col].value_counts()) == 1:
            constant_columns.append(col)

    # categorical featuresの処理
    cat_cols = [c for c in df.columns if not c.startswith("total")]
    cat_cols = [c for c in cat_cols if c not in constant_columns + EXCLUDED_FEATURES]

    # cat colsはとりあえずlabel encodingのみ
    for c in cat_cols:
        le = LabelEncoder()
        df_vals = list(df[c].values.astype(str))
        le.fit(df_vals)
        df[c] = le.transform(df_vals)

    # numeric featuresの処理
    num_cols = [c for c in df.columns if c.startswith("total")]
    num_cols = [c for c in num_cols if c not in constant_columns + EXCLUDED_FEATURES]

    # 数値型へ変換
    df[num_cols] = df[num_cols].astype(float)

    # target変数をfillnaしておきます
    df["totals.transactionRevenue"] = df["totals.transactionRevenue"].fillna(0.0)

    df = df[cat_cols+num_cols+['IS_TEST']]
    return df, cat_cols

if __name__ == '__main__':
    # test
    df, cat = get_df(num_rows=10000)
    df.to_csv("df_test.csv")
    print(df)
