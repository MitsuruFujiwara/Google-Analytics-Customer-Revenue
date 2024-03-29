
import os
import json
import pandas as pd
import numpy as np
import time
import pickle
import logging

from glob import glob
from time import time, sleep
from pandas.io.json import json_normalize
from multiprocessing import Pool as Pool
from contextlib import contextmanager
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm

import requests
import functools
import gc

'''
Utility的なものを置いとくところ
'''

NUM_FOLDS = 10

COMPETITION_NAME = 'ga-customer-revenue-prediction'

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
                     "visitStartTime", 'index', 'IS_TEST', 'TARGET_BIN', 'vis_date']
excluded_features_v2 = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue',
    'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits'
]

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

# correlationの高い変数を削除する機能
def removeCorrelatedVariables(data, threshold):
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    col_drop = [column for column in upper.columns if any(upper[column] > threshold) & ('totals.transactionRevenue' not in column)]
    return col_drop

# 欠損値の率が高い変数を削除する機能
def removeMissingVariables(data, threshold):
    missing = (data.isnull().sum() / len(data)).sort_values(ascending = False)
    col_missing = missing.index[missing > threshold]
    col_missing = [column for column in col_missing if 'totals.transactionRevenue' not in column]
    return col_missing

def save2pkl(path, df):
    f = open(path, 'wb')
    pickle.dump(df, f)
    f.close

def loadpkl(path):
    f = open(path, 'rb')
    out = pickle.load(f)
    return out

# 拾い物 https://www.kaggle.com/yoshoku/gacrp-v2-starter-kit/code
def load_df(csv_path, num_rows=None, chunksize=100000):
    features = ['channelGrouping', 'date', 'fullVisitorId', 'visitId',
                'visitNumber', 'visitStartTime', 'device.browser',
                'device.deviceCategory', 'device.isMobile', 'device.operatingSystem',
                'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country',
                'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region',
                'geoNetwork.subContinent', 'totals.bounces', 'totals.hits',
                'totals.newVisits', 'totals.pageviews', 'totals.transactionRevenue',
                'trafficSource.adContent', 'trafficSource.campaign',
                'trafficSource.isTrueDirect', 'trafficSource.keyword',
                'trafficSource.medium', 'trafficSource.referralPath',
                'trafficSource.source']
    JSON_COLS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    print('Load {}'.format(csv_path))
    df_reader = pd.read_csv(csv_path,
                            converters={ column: json.loads for column in JSON_COLS },
                            dtype={ 'date': str, 'fullVisitorId': str, 'sessionId': str },
                            chunksize=chunksize, nrows=num_rows)
    res = pd.DataFrame()
    for cidx, df in enumerate(df_reader):
        df.reset_index(drop=True, inplace=True)
        for col in JSON_COLS:
            col_as_df = json_normalize(df[col])
            col_as_df.columns = ['{}.{}'.format(col, subcol) for subcol in col_as_df.columns]
            df = df.drop(col, axis=1).merge(col_as_df, right_index=True, left_index=True)
        res = pd.concat([res, df[features]], axis=0).reset_index(drop=True)
        del df
        gc.collect()
        print('{}: {}'.format(cidx + 1, res.shape))
    return res

def line_notify(message):
    f = open('../input/line_token.txt')
    token = f.read()
    f.close
    line_notify_token = token.replace('\n', '')
    line_notify_api = 'https://notify-api.line.me/api/notify'

    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}  # 発行したトークン
    line_notify = requests.post(line_notify_api, data=payload, headers=headers)
    print(message)

# API経由でsubmitする機能 https://github.com/KazukiOnodera/Home-Credit-Default-Risk/blob/master/py/utils.py
def submit(file_path, comment='from API'):
    os.system('kaggle competitions submit -c {} -f {} -m "{}"'.format(COMPETITION_NAME,file_path,comment))
    sleep(60) # tekito~~~~
    tmp = os.popen('kaggle competitions submissions -c {} -v | head -n 2'.format(COMPETITION_NAME)).read()
    col, values = tmp.strip().split('\n')
    message = 'SCORE!!!\n'
    for i,j in zip(col.split(','), values.split(',')):
        message += '{}: {}\n'.format(i,j)
#        print(f'{i}: {j}') # TODO: comment out later?
    line_notify(message.rstrip())

def develop_json_fields(df=None):
    json_fields = ['device', 'geoNetwork', 'totals', 'trafficSource']
    # Get the keys
    for json_field in json_fields:
        # print('Doing Field {}'.format(json_field))
        # Get json field keys to create columns
        the_keys = get_keys_for_field(json_field)
        # Replace the string by a dict
        # print('Transform string to dict')
        df[json_field] = multi_apply_func_on_series(
            df=df[json_field],
            func=convert_to_dict,
            n_jobs=4
        )
        #logger.info('{} converted to dict'.format(json_field))
        #         df[json_field] = df[json_field].apply(lambda x: eval(x
        #                                             .replace('false', 'False')
        #                                             .replace('true', 'True')
        #                                             .replace('null', 'np.nan')))
        for k in the_keys:
            # print('Extracting {}'.format(k))
            df[json_field + '.' + k] = df[json_field].apply(lambda x: get_dict_field(x_=x, key_=k))
        del df[json_field]
        gc.collect()
        #logger.info('{} fields extracted'.format(json_field))
    return df

def get_keys_for_field(field=None):
    the_dict = {
        'device': [
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

    return the_dict[field]

def try_encode(x):
    """Used to remove any encoding issues within the data"""
    try:
        return x.encode('utf-8', 'surrogateescape').decode('utf-8')
    except AttributeError:
        return np.nan
    except UnicodeEncodeError:
        return np.nan

def browser_mapping(x):
    browsers = ['chrome','safari','firefox','internet explorer','edge','opera','coc coc','maxthon','iron']
    if x in browsers:
        return x.lower()
    elif  ('android' in x) or ('samsung' in x) or ('mini' in x) or ('iphone' in x) or ('in-app' in x) or ('playstation' in x):
        return 'mobile browser'
    elif  ('mozilla' in x) or ('chrome' in x) or ('blackberry' in x) or ('nokia' in x) or ('browser' in x) or ('amazon' in x):
        return 'mobile browser'
    elif  ('lunascape' in x) or ('netscape' in x) or ('blackberry' in x) or ('konqueror' in x) or ('puffin' in x) or ('amazon' in x):
        return 'mobile browser'
    elif '(not set)' in x:
        return x
    else:
        return 'others'


def adcontents_mapping(x):
    if  ('google' in x):
        return 'google'
    elif  ('placement' in x) | ('placememnt' in x):
        return 'placement'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'ad' in x:
        return 'ad'
    else:
        return 'others'

def source_mapping(x):
    if  ('google' in x):
        return 'google'
    elif  ('youtube' in x):
        return 'youtube'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'yahoo' in x:
        return 'yahoo'
    elif 'facebook' in x:
        return 'facebook'
    elif 'reddit' in x:
        return 'reddit'
    elif 'bing' in x:
        return 'bing'
    elif 'quora' in x:
        return 'quora'
    elif 'outlook' in x:
        return 'outlook'
    elif 'linkedin' in x:
        return 'linkedin'
    elif 'pinterest' in x:
        return 'pinterest'
    elif 'ask' in x:
        return 'ask'
    elif 'siliconvalley' in x:
        return 'siliconvalley'
    elif 'lunametrics' in x:
        return 'lunametrics'
    elif 'amazon' in x:
        return 'amazon'
    elif 'mysearch' in x:
        return 'mysearch'
    elif 'qiita' in x:
        return 'qiita'
    elif 'messenger' in x:
        return 'messenger'
    elif 'twitter' in x:
        return 'twitter'
    elif 't.co' in x:
        return 't.co'
    elif 'vk.com' in x:
        return 'vk.com'
    elif 'search' in x:
        return 'search'
    elif 'edu' in x:
        return 'edu'
    elif 'mail' in x:
        return 'mail'
    elif 'ad' in x:
        return 'ad'
    elif 'golang' in x:
        return 'golang'
    elif 'direct' in x:
        return 'direct'
    elif 'dealspotr' in x:
        return 'dealspotr'
    elif 'sashihara' in x:
        return 'sashihara'
    elif 'phandroid' in x:
        return 'phandroid'
    elif 'baidu' in x:
        return 'baidu'
    elif 'mdn' in x:
        return 'mdn'
    elif 'duckduckgo' in x:
        return 'duckduckgo'
    elif 'seroundtable' in x:
        return 'seroundtable'
    elif 'metrics' in x:
        return 'metrics'
    elif 'sogou' in x:
        return 'sogou'
    elif 'businessinsider' in x:
        return 'businessinsider'
    elif 'github' in x:
        return 'github'
    elif 'gophergala' in x:
        return 'gophergala'
    elif 'yandex' in x:
        return 'yandex'
    elif 'msn' in x:
        return 'msn'
    elif 'dfa' in x:
        return 'dfa'
    elif '(not set)' in x:
        return '(not set)'
    elif 'feedly' in x:
        return 'feedly'
    elif 'arstechnica' in x:
        return 'arstechnica'
    elif 'squishable' in x:
        return 'squishable'
    elif 'flipboard' in x:
        return 'flipboard'
    elif 't-online.de' in x:
        return 't-online.de'
    elif 'sm.cn' in x:
        return 'sm.cn'
    elif 'wow' in x:
        return 'wow'
    elif 'baidu' in x:
        return 'baidu'
    elif 'partners' in x:
        return 'partners'
    else:
        return 'others'

def process_device(data_df):
    print("process device ...")
    data_df['source.country'] = data_df['trafficSource.source'] + '_' + data_df['geoNetwork.country']
    data_df['campaign.medium'] = data_df['trafficSource.campaign'] + '_' + data_df['trafficSource.medium']
    data_df['browser.category'] = data_df['device.browser'] + '_' + data_df['device.deviceCategory']
    data_df['browser.os'] = data_df['device.browser'] + '_' + data_df['device.operatingSystem']
    return data_df

def custom(data):
    print('custom..')
    data['device_deviceCategory_channelGrouping'] = data['device.deviceCategory'] + "_" + data['channelGrouping']
    data['channelGrouping_browser'] = data['device.browser'] + "_" + data['channelGrouping']
    data['channelGrouping_OS'] = data['device.operatingSystem'] + "_" + data['channelGrouping']

    for i in ['geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country','geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region','geoNetwork.subContinent']:
        for j in ['device.browser','device.deviceCategory', 'device.operatingSystem', 'trafficSource.source']:
            data[i + "_" + j] = data[i] + "_" + data[j]

    data['content.source'] = data['trafficSource.adContent'] + "_" + data['source.country']
    data['medium.source'] = data['trafficSource.medium'] + "_" + data['source.country']
    return data

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mkdir_p(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)

# 複数のpklファイルに分割して保存する機能
def to_pickles(df, path, split_size=3, inplace=True):
    """
    path = '../output/mydf'

    wirte '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'
    """
    print('shape: {}'.format(df.shape))

#    if inplace==True:
#        df.reset_index(drop=True, inplace=True)
#    else:
#        df = df.reset_index(drop=True)
    gc.collect()
    mkdir_p(path)

    kf = KFold(n_splits=split_size)
    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
        df.iloc[val_index].to_pickle(path+'/'+str(i)+'.pkl')
    return

# path以下の複数pklファイルを読み込む機能
def read_pickles(path, col=None, use_tqdm=True):
    if col is None:
        if use_tqdm:
            df = pd.concat([ pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*'))) ])
        else:
            print('reading {}'.format(path))
            df = pd.concat([ pd.read_pickle(f) for f in sorted(glob(path+'/*')) ])
    else:
        df = pd.concat([ pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*'))) ])
    return df
