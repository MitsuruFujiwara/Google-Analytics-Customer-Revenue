import pandas as pd
import gc
import numpy as np

from sklearn.preprocessing import LabelEncoder
from Utils import load_df, EXCLUDED_FEATURES, one_hot_encoder

# columns毎にtarget encodingを適用する関数
def targetEncoding(df, col, target='TARGET_BIN'):
    dict_for_map = df[~df['IS_TEST']].fillna(-1).groupby(col)[target].mean()
    res = df[col].fillna(-1).map(dict_for_map)
    return res

def get_df(num_rows=None):
    print("Loading datasets...")
    # load datasets
    train_df = load_df('../input/train.csv', nrows=num_rows)
    test_df = load_df('../input/test.csv', nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))

    # train testの識別用
    train_df['IS_TEST'] = False
    test_df['IS_TEST'] = True

    # leak dataの取得
    train_store_1 = pd.read_csv('../input/Train_external_data.csv',
                                low_memory=False, skiprows=6, dtype={"Client Id":'str'})
    train_store_2 = pd.read_csv('../input/Train_external_data_2.csv',
                                low_memory=False, skiprows=6, dtype={"Client Id":'str'})
    test_store_1 = pd.read_csv('../input/Test_external_data.csv',
                                low_memory=False, skiprows=6, dtype={"Client Id":'str'})
    test_store_2 = pd.read_csv('../input/Test_external_data_2.csv',
                                low_memory=False, skiprows=6, dtype={"Client Id":'str'})

    leak_cols = train_store_1.columns.tolist()

    # Getting VisitId from Google Analytics...
    for df in [train_store_1, train_store_2, test_store_1, test_store_2]:
        df["visitId"] = df["Client Id"].apply(lambda x: x.split('.', 1)[1]).astype(np.int64)

    # Merge with train/test data
    train_df = train_df.merge(pd.concat([train_store_1, train_store_2], sort=False), how="left", on="visitId")
    test_df = test_df.merge(pd.concat([test_store_1, test_store_2], sort=False), how="left", on="visitId")

    # Merge
    df = train_df.append(test_df).reset_index()

    # Cleaning Revenue
    df["Revenue"].fillna('$', inplace=True)
    df["Revenue"] = df["Revenue"].apply(lambda x: x.replace('$', '').replace(',', ''))
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df[leak_cols] = df[leak_cols].fillna(0.0)

    # Clearing leaked data:
    df["Avg. Session Duration"][df["Avg. Session Duration"] == 0] = "00:00:00"
    df["Avg. Session Duration"] = df["Avg. Session Duration"].str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]))
    df["Bounce Rate"] = df["Bounce Rate"].astype(str).apply(lambda x: x.replace('%', '')).astype(float)
    df["Goal Conversion Rate"] = df["Goal Conversion Rate"].astype(str).apply(lambda x: x.replace('%', '')).astype(float)

    # drop Client Id
    df.drop("Client Id", 1, inplace=True)

    del train_df, test_df
    gc.collect()

    # 不要カラムを抽出
    for col in df.columns:
        if len(df[col].value_counts()) == 1:
            EXCLUDED_FEATURES.append(col)

    # 季節性変数の処理
    df['vis_date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['dayofweek'] = df['vis_date'].dt.dayofweek
    df['hour'] = df['vis_date'].dt.hour
    df['day'] = df['vis_date'].dt.day
    df['month'] = df['vis_date'].dt.month
    df['weekday'] = df['vis_date'].dt.weekday
    df['time'] = df['vis_date'].dt.second + df['vis_date'].dt.minute*60 + df['vis_date'].dt.hour*3600

    # remember these features were equal, but not always? May be it means something...
    df["id_incoherence"] = pd.to_datetime(df.visitId, unit='s') != df['vis_date']
    # remember visitId dublicates?
    df["visitId_dublicates"] = df.visitId.map(df.visitId.value_counts())
    # remember session dublicates?
    df["session_dublicates"] = df.sessionId.map(df.sessionId.value_counts())

    # paired categories
    df['source.country'] = df['trafficSource.source'] + '_' + df['geoNetwork.country']
    df['campaign.medium'] = df['trafficSource.campaign'] + '_' + df['trafficSource.medium']
    df['browser.category'] = df['device.browser'] + '_' + df['device.deviceCategory']
    df['browser.os'] = df['device.browser'] + '_' + df['device.operatingSystem']

    df['device_deviceCategory_channelGrouping'] = df['device.deviceCategory'] + "_" + df['channelGrouping']
    df['channelGrouping_browser'] = df['device.browser'] + "_" + df['channelGrouping']
    df['channelGrouping_OS'] = df['device.operatingSystem'] + "_" + df['channelGrouping']

    for i in ['geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country','geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region','geoNetwork.subContinent']:
        for j in ['device.browser','device.deviceCategory', 'device.operatingSystem', 'trafficSource.source']:
            df[i + "_" + j] = df[i] + "_" + df[j]

    df['content.source'] = df['trafficSource.adContent'].astype(str) + "_" + df['source.country']
    df['medium.source'] = df['trafficSource.medium'] + "_" + df['source.country']

    # User-aggregating features
    df[["totals.hits", "totals.pageviews", "visitNumber"]] = df[["totals.hits", "totals.pageviews", "visitNumber"]].fillna(0)
    df[["totals.hits", "totals.pageviews", "visitNumber"]] = df[["totals.hits", "totals.pageviews", "visitNumber"]].astype(int)
    for feature in ["totals.hits", "totals.pageviews"]:
        info = df.groupby("fullVisitorId")[feature].mean()
        df["usermean_" + feature] = df.fullVisitorId.map(info)

    for feature in ["visitNumber"]:
        info = df.groupby("fullVisitorId")[feature].max()
        df["usermax_" + feature] = df.fullVisitorId.map(info)

    # categorical featuresの処理
    cat_cols = [c for c in df.columns if not c.startswith("total") and c not in EXCLUDED_FEATURES+leak_cols+['time']]

    # target encoding用のラベルを生成
    df['TARGET_BIN'] = df['totals.transactionRevenue'].notnull()*1

    # target encoding
    for c in cat_cols:
        print("target encoding: {}".format(c))
        df[c] = targetEncoding(df, c, target='TARGET_BIN')

    # numeric columnsの抽出
    num_cols = [c for c in df.columns if c.startswith("total") and c not in EXCLUDED_FEATURES+leak_cols]
    num_cols = num_cols+["Revenue", 'time', "Avg. Session Duration", "Bounce Rate", "Goal Conversion Rate"]

    # numeric columnsを数値型へ変換
    df[num_cols] = df[num_cols].astype(float)

    # fillna
    df[num_cols] = df[num_cols].fillna(0)

    # future features
    df.sort_values(['fullVisitorId', 'date'], ascending=True, inplace=True)
    df['prev_session'] = (df['vis_date'] - df[['fullVisitorId', 'vis_date']].groupby('fullVisitorId')['vis_date'].shift(1)
                            ).astype(np.int64) // 1e9 // 60 // 60
    df['next_session'] = (df['vis_date'] - df[['fullVisitorId', 'vis_date']].groupby('fullVisitorId')['vis_date'].shift(-1)
                            ).astype(np.int64) // 1e9 // 60 // 60
    df.sort_index(inplace=True)

    """
    df.sort_values(['fullVisitorId', 'vis_date'], ascending=True, inplace=True)
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

#    df = df[cat_cols+num_cols+['IS_TEST']]
    return df

if __name__ == '__main__':
    # test
    df = get_df(num_rows=10000)
    print(df)
