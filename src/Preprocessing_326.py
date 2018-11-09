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
    train_df = load_df('../input/train_v2.csv', nrows=num_rows)
    test_df = load_df('../input/test_v2.csv', nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))

    # train testの識別用
    train_df['IS_TEST'] = False
    test_df['IS_TEST'] = True

    # 外れ値の処理
#    train_df.loc[:,'totals.transactionRevenue'] = train_df['totals.transactionRevenue'].astype(float).fillna(0)
#    mean = train_df[train_df['totals.transactionRevenue']>0]['totals.transactionRevenue'].mean()
#    std = train_df[train_df['totals.transactionRevenue']>0]['totals.transactionRevenue'].std()
#    threshold =  mean + std*2
#    print("mean: {}, std: {}, threshold: {}".format(mean, std, threshold))
#    train_df = train_df[train_df['totals.transactionRevenue'] < threshold]

    # Merge
    df = train_df.append(test_df).reset_index()

    del train_df, test_df
    gc.collect()

    # 不要カラムを抽出
    for col in df.columns:
        if len(df[col].value_counts()) == 1:
            EXCLUDED_FEATURES.append(col)

    # 季節性変数の処理
    df['vis_date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['hour'] = df['vis_date'].dt.hour
    df['day'] = df['vis_date'].dt.day
    df['weekday'] = df['vis_date'].dt.weekday
    df['month'] = df['vis_date'].dt.month
    df['weekofyear'] = df['vis_date'].dt.weekofyear

    df['hour_day'], _ = pd.factorize(df['hour'].astype(str)+'_'+df['day'].astype(str))
    df['hour_weekday'], _ = pd.factorize(df['hour'].astype(str)+'_'+df['weekday'].astype(str))
    df['hour_month'], _ = pd.factorize(df['hour'].astype(str)+'_'+df['month'].astype(str))
    df['hour_weekofyear'], _ = pd.factorize(df['hour'].astype(str)+'_'+df['weekofyear'].astype(str))
    df['day_weekday'], _ = pd.factorize(df['day'].astype(str)+'_'+df['weekday'].astype(str))
    df['day_month'], _ = pd.factorize(df['day'].astype(str)+'_'+df['month'].astype(str))
    df['day_weekofyear'], _ = pd.factorize(df['day'].astype(str)+'_'+df['weekofyear'].astype(str))
    df['weekday_month'], _ = pd.factorize(df['weekday'].astype(str)+'_'+df['month'].astype(str))

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

    # categorical featuresの処理
    cat_cols = [c for c in df.columns if not c.startswith("total") and c not in EXCLUDED_FEATURES+['time']]

    # target encoding用のラベルを生成
    df['TARGET_BIN'] = df['totals.transactionRevenue'].notnull()*1

    # target encoding
    for c in cat_cols:
        print("target encoding: {}".format(c))
        df[c] = targetEncoding(df, c, target='TARGET_BIN')

    # numeric columnsの抽出
    num_cols = [c for c in df.columns if c.startswith("total") and c not in EXCLUDED_FEATURES]

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

    # User-aggregating features
    df[["totals.hits", "totals.pageviews", "visitNumber"]] = df[["totals.hits", "totals.pageviews", "visitNumber"]].fillna(0)
    df[["totals.hits", "totals.pageviews", "visitNumber"]] = df[["totals.hits", "totals.pageviews", "visitNumber"]].astype(int)
    for feature in ["totals.hits", "totals.pageviews"]:
        info = df.groupby("fullVisitorId")[feature].mean()
        df["usermean_" + feature] = df.fullVisitorId.map(info)

    for feature in ["visitNumber"]:
        info = df.groupby("fullVisitorId")[feature].max()
        df["usermax_" + feature] = df.fullVisitorId.map(info)

    #当日のhit数が全体に占める割合
    df.loc[:,'totals.hits'] = df.loc[:,'totals.hits'].astype('float64')
    df['nb_hits'] = df['date'].map(df[['date', 'totals.hits']].groupby('date')['totals.hits'].sum())
    df['ratio_hits'] = df['totals.hits'] / df['nb_hits']

    #当日のpv数が全体に占める割合
    df.loc[:,'totals.pageviews'] = df.loc[:,'totals.pageviews'].astype('float64')
    df['nb_pageviews'] = df['date'].map(df[['date', 'totals.pageviews']].groupby('date')['totals.pageviews'].sum())
    df['ratio_pageviews'] = df['totals.pageviews'] / df['nb_pageviews']

    # 当日のvisit number数が全体に占める割合
    df.loc[:,'visitNumber'] = df.loc[:,'visitNumber'].astype('float64')
    df['nb_visitNumber'] = df['date'].map(df[['date', 'visitNumber']].groupby('date')['visitNumber'].sum())
    df['ratio_visitNumber'] = df['visitNumber'] / df['nb_visitNumber']

    return df

if __name__ == '__main__':
    # test
    df = get_df(num_rows=10000)
    print(df)
