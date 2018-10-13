
import json
import pandas as pd
import numpy as np
import time
import pickle
import logging

from pandas.io.json import json_normalize
from multiprocessing import Pool as Pool
from contextlib import contextmanager
import requests
import functools
import gc

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