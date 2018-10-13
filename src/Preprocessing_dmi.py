import pandas as pd
import gc
import numpy as np

from sklearn.preprocessing import LabelEncoder
from Utils import load_df, excluded_features_v2, develop_json_fields, try_encode, browser_mapping, \
                  adcontents_mapping, source_mapping, process_device, custom


def get_df_2(nrows=None):
    # Convert train
    train = pd.read_csv('../input/train.csv', dtype='object', nrows=nrows, encoding='utf-8')
    train = develop_json_fields(df=train)
    #logger.info('Train done')
    print('train done')

    # Convert test
    test = pd.read_csv('../input/test.csv', dtype='object', nrows=nrows, encoding='utf-8')
    test = develop_json_fields(df=test)
    #logger.info('Test done')
    print('test done')

    # Check features validity
    for f in train.columns:
        if f not in ['date', 'fullVisitorId', 'sessionId']:
            try:
                train[f] = train[f].astype(np.float64)
                test[f] = test[f].astype(np.float64)
            except (ValueError, TypeError):
                #logger.info('{} is a genuine string field'.format(f))
                print('{} is a genuine string field'.format(f))
                pass
            except Exception:
                #logger.exception('{} enountered an exception'.format(f))
                print('{} enountered an exception'.format(f))
                raise

    #logger.info('{}'.format(train['totals.transactionRevenue'].sum()))
    feature_to_drop = []
    for f in train.columns:
        if f not in ['date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue']:
            if train[f].dtype == 'object':
                try:
                    trn, _ = pd.factorize(train[f])
                    tst, _ = pd.factorize(test[f])
                    if (np.std(trn) == 0) | (np.std(tst) == 0):
                        feature_to_drop.append(f)
                        #logger.info('No variation in {}'.format(f))
                except TypeError:
                    feature_to_drop.append(f)
                    #logger.info('TypeError exception for {}'.format(f))
            else:
                if (np.std(train[f].fillna(0).values) == 0) | (np.std(test[f].fillna(0).values) == 0):
                    feature_to_drop.append(f)
                    #logger.info('No variation in {}'.format(f))
    test.drop(feature_to_drop, axis=1, inplace=True)
    train.drop(feature_to_drop, axis=1, inplace=True)
    #logger.info('{}'.format(train['totals.transactionRevenue'].sum()))

    for f in train.columns:
        if train[f].dtype == 'object':
            train[f] = train[f].apply(lambda x: try_encode(x))
            test[f] = test[f].apply(lambda x: try_encode(x))
            #cat_cols.append(f)

    temp = list(train[train.loc[:,'totals.transactionRevenue']>0].loc[:,'totals.transactionRevenue'])
    v = np.mean(temp)+np.std(temp)*2

    train['totals.transactionRevenue'] = train['totals.transactionRevenue'].fillna(0)
    train = train[train.loc[:,'totals.transactionRevenue']<v]
    y_reg = train['totals.transactionRevenue']
    del train['totals.transactionRevenue']

    if 'totals.transactionRevenue' in test.columns:
        del test['totals.transactionRevenue']
        
    #train['target'] = y_reg
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

        df['device.browser'] = df['device.browser'].map(lambda x:browser_mapping(str(x).lower())).astype('str')
        df['trafficSource.adContent'] = df['trafficSource.adContent'].map(lambda x:adcontents_mapping(str(x).lower())).astype('str')
        df['trafficSource.source'] = df['trafficSource.source'].map(lambda x:source_mapping(str(x).lower())).astype('str')

        df = process_device(df)
        df = custom(df)
        
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

    categorical_features = [
        _f for _f in train.columns
        if (_f not in excluded_features_v2) & (train[_f].dtype == 'object')
    ]

    for f in categorical_features:
        train[f], indexer = pd.factorize(train[f])
        test[f] = indexer.get_indexer(test[f])

    #y_reg = train['target']
    #del train['target']

    test.to_csv('../input/extracted_fields_test.csv', index=False)
    train.to_csv('../input/extracted_fields_train.csv', index=False)

    #train['IS_TEST'] = False
    #test['IS_TEST'] = True

    return train, test

"""
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
"""

if __name__ == '__main__':
    # test
    train, test  = get_df_2(nrows=10000)
    #df.to_csv("df_test.csv")
    print(len(train.columns))
