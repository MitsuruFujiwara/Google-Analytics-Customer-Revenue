
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
import gc
import time
import warnings

from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning

from Utils import line_notify, submit, rmse
from Utils import EXCLUDED_FEATURES, NUM_FOLDS
from Preprocessing_326 import get_df

################################################################################
# 複数モデルのoutputをブレンドして最終的なsubmitファイルを生成するスクリプト。
################################################################################

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# LightGBM
def predict_lgbm(df, submission_file_name_lgbm):
    # Divide in training/validation and test data
    test_df = df[df['IS_TEST']]

    del df
    gc.collect()

    ############################################################################
    # Session Level predictions
    ############################################################################

    print('Starting Session Level predictions...')

    sub_preds_session = np.zeros(test_df.shape[0])
    feats = [f for f in test_df.columns if f not in EXCLUDED_FEATURES+['totals.transactionRevenue']]

    for n_fold in range(NUM_FOLDS):
        # load model
        reg = lgb.Booster(model_file='../output/models/lgbm_session_'+str(n_fold)+'.txt')

        # prediction
        sub_preds_session += np.expm1(reg.predict(test_df[feats], num_iteration=reg.best_iteration)) / NUM_FOLDS

        print('Session Level prediction fold {} done.'.format(n_fold+1))

        del reg
        gc.collect()

    # add predictions column
    test_df.loc[:,'predictions'] = sub_preds_session

    ############################################################################
    # User Level predictions
    ############################################################################

    print('Starting User Level predictions...')

    # Aggregate data at User level
    aggregations = {'totals.transactionRevenue': ['sum']}
    for col in feats+['predictions']:
        aggregations[col] = ['sum', 'max', 'min', 'mean']

    test_df_agg = test_df[feats + ['fullVisitorId','totals.transactionRevenue', 'predictions']].groupby('fullVisitorId').agg(aggregations)
    del test_df
    gc.collect()

    # reshape header
    test_df_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in test_df_agg.columns.tolist()])

    # to float32
    test_df_agg=test_df_agg.astype('float32')

    sub_preds_agg = np.zeros(test_df_agg.shape[0])
    feats_agg = [f for f in test_df_agg.columns if f not in EXCLUDED_FEATURES+['totals.transactionRevenue_SUM']]

    for n_fold in range(NUM_FOLDS):
        # load model
        reg = lgb.Booster(model_file='../output/models/lgbm_user_'+str(n_fold)+'.txt')

        # prediction
        sub_preds_agg += np.expm1(reg.predict(test_df_agg[feats_agg], num_iteration=reg.best_iteration)) / NUM_FOLDS

        print('User Level prediction fold {} done.'.format(n_fold+1))

        del reg
        gc.collect()

    # 提出データの予測値を保存
    test_df_agg.loc[:,'PredictedLogRevenue'] = sub_preds_agg
    submission = test_df_agg[['PredictedLogRevenue']]
    submission['PredictedLogRevenue'] = np.log1p(submission['PredictedLogRevenue'])
    submission['PredictedLogRevenue'] =  submission['PredictedLogRevenue'].apply(lambda x : 0.0 if x < 0 else x)
    submission['PredictedLogRevenue'] = submission['PredictedLogRevenue'].fillna(0)
    submission.to_csv(submission_file_name_lgbm , index=True)

    # API経由でsubmit
    submit(submission_file_name_lgbm, comment='LightGBM再現性テスト')

    return submission

# XGBoost
def predict_xgb(df, submission_file_name_xgb):
    # Divide in training/validation and test data
    test_df = df[df['IS_TEST']]

    del df
    gc.collect()

    ############################################################################
    # Session Level predictions
    ############################################################################

    print('Starting Session Level predictions...')

    sub_preds_session = np.zeros(test_df.shape[0])
    feats = [f for f in test_df.columns if f not in EXCLUDED_FEATURES+['totals.transactionRevenue']]

    test_df_dmtrx = xgb.DMatrix(test_df[feats], label=train_df['totals.transactionRevenue'])

    for n_fold in range(NUM_FOLDS):
        # load model
        reg = xgb.Booster(model_file='../output/models/xgb_session_'+str(n_fold)+'.txt')

        # prediction
        sub_preds_session += np.expm1(reg.predict(test_df_dmtrx)) / NUM_FOLDS

        print('Session Level prediction fold {} done.'.format(n_fold+1))

        del reg
        gc.collect()

    # add predictions column
    test_df.loc[:,'predictions'] = sub_preds_session

    ############################################################################
    # User Level predictions
    ############################################################################

    print('Starting User Level predictions...')

    # Aggregate data at User level
    aggregations = {'totals.transactionRevenue': ['sum']}
    for col in feats+['predictions']:
        aggregations[col] = ['sum', 'max', 'min', 'mean']

    test_df_agg = test_df[feats + ['fullVisitorId','totals.transactionRevenue', 'predictions']].groupby('fullVisitorId').agg(aggregations)
    del test_df
    gc.collect()

    # reshape header
    test_df_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in test_df_agg.columns.tolist()])

    # to float32
    test_df_agg=test_df_agg.astype('float32')

    sub_preds_agg = np.zeros(test_df_agg.shape[0])
    feats_agg = [f for f in test_df_agg.columns if f not in EXCLUDED_FEATURES+['totals.transactionRevenue_SUM']]

    test_df_dmtrx_agg = xgb.DMatrix(test_df_agg[feats_agg], label=test_df_agg['totals.transactionRevenue_SUM'])

    for n_fold in range(NUM_FOLDS):
        # load model
        reg = xgb.Booster(model_file='../output/models/xgb_user_'+str(n_fold)+'.txt')

        # prediction
        sub_preds_agg += np.expm1(reg.predict(test_df_dmtrx_agg)) / NUM_FOLDS

        print('User Level prediction fold {} done.'.format(n_fold+1))

        del reg
        gc.collect()

    # 提出データの予測値を保存
    test_df_agg.loc[:,'PredictedLogRevenue'] = sub_preds_agg
    submission = test_df_agg[['PredictedLogRevenue']]
    submission['PredictedLogRevenue'] = np.log1p(submission['PredictedLogRevenue'])
    submission['PredictedLogRevenue'] =  submission['PredictedLogRevenue'].apply(lambda x : 0.0 if x < 0 else x)
    submission['PredictedLogRevenue'] = submission['PredictedLogRevenue'].fillna(0)
    submission.to_csv(submission_file_name_xgb , index=True)

    # API経由でsubmit
    submit(submission_file_name_xgb, comment='XGBoost再現性テスト')

    return submission

def main():
    # submitファイルをロード
    sub = pd.read_csv("../input/sample_submission_v2.csv",dtype={'fullVisitorId': str})
    sub_lgbm = pd.read_csv("../output/submission_lgbm.csv",dtype={'fullVisitorId': str})
    sub_xgb = pd.read_csv("../output/submission_xgb.csv",dtype={'fullVisitorId': str})

    # merge
    sub['lgbm'] = np.expm1(sub_lgbm['PredictedLogRevenue'])
    sub['xgb'] = np.expm1(sub_xgb['PredictedLogRevenue'])
    sub.loc[:,'PredictedLogRevenue'] = np.log1p(0.5*sub['lgbm']+0.5*sub['xgb'])

    del sub_lgbm, sub_xgb
    gc.collect()

    # out of foldの予測値をロード
    oof_lgbm = pd.read_csv("../output/oof_lgbm.csv",dtype={'fullVisitorId': str})
    oof_xgb = pd.read_csv("../output/oof_xgb.csv",dtype={'fullVisitorId': str})
    oof = 0.5*oof_lgbm['OOF_PRED']+0.5*oof_xgb['OOF_PRED']

    # local cv scoreを算出
    local_rmse = rmse(np.log1p(oof_lgbm['totals.transactionRevenue_SUM']), np.log1p(oof))

    del oof_lgbm, oof_xgb
    gc.collect()

    # save submit file
    sub[['fullVisitorId', 'PredictedLogRevenue']].to_csv(submission_file_name, index=False)

    # submit
    submit(submission_file_name, comment='cv: %.6f' % local_rmse)

if __name__ == '__main__':
    submission_file_name = "../output/submission_blend.csv"
    submission_file_name_lgbm ="../output/submission_lgbm_pred.csv"
    submission_file_name_xgb ="../output/submission_xgb_pred.csv"
    """
    with timer("Preprocessing"):
        df = get_df(num_rows=None)
    with timer("LightGBM prediction"):
        predict_lgbm(df, submission_file_name_lgbm)
    with timer("XGBoost prediction"):
        predict_lgbm(df, submission_file_name_xgb)
    """
    with timer("Blend prediction"):
        main()
