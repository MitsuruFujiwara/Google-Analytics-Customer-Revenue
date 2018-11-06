
import lightgbm as lgb
import numpy as np
import pandas as pd
import gc
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from contextlib import contextmanager
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from pandas.core.common import SettingWithCopyWarning

from Preprocessing_326 import get_df
from Utils import EXCLUDED_FEATURES, line_notify, submit, NUM_FOLDS

################################################################################
# Kuso-simple LightGBM k-fold
# とりあえずSantanderの改変です
################################################################################

warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Define folding strategy¶
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

# Display/plot feature importance
def display_importances(feature_importance_df_, outputpath, csv_outputpath):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    # importance下位の確認用に追加しました
    _feature_importance_df_=feature_importance_df_.groupby('feature').sum()
    _feature_importance_df_.to_csv(csv_outputpath)

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outputpath)

# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):

    # Divide in training/validation and test data
    train_df = df[~df['IS_TEST']]
    test_df = df[df['IS_TEST']]

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()

    ############################################################################
    # Session Level predictions
    ############################################################################

    # Cross validation model
    folds_session = get_folds(df=train_df, n_splits=num_folds)

    # Create arrays and dataframes to store results
    oof_preds_session = np.zeros(train_df.shape[0])
    sub_preds_session = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in EXCLUDED_FEATURES+['totals.transactionRevenue']]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds_session):
        train_x, train_y = train_df[feats].iloc[train_idx], np.log1p(train_df['totals.transactionRevenue'].iloc[train_idx])
        valid_x, valid_y = train_df[feats].iloc[valid_idx], np.log1p(train_df['totals.transactionRevenue'].iloc[valid_idx])

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # パラメータは適当です
        params ={
                'device' : 'gpu',
#                'gpu_use_dp':True,
                'task': 'train',
                'boosting': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'num_iteration': 10000,
                'learning_rate': 0.02,
                'num_leaves': 64,
                'colsample_bytree': 0.553240348074409,
                'subsample': 0.471522873020333,
                'max_depth': 8,
                'reg_alpha': 9.83318745912308,
                'reg_lambda': 0.925142409255232,
                'min_split_gain': 0.954402595384603,
                'min_child_weight': 44,
                'min_data_in_leaf': 79,
                'verbose': -1,
                'seed':int(2**n_fold),
                'bagging_seed':int(2**n_fold),
                'drop_seed':int(2**n_fold)
                }

        reg = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        oof_preds_session[valid_idx] = np.expm1(reg.predict(valid_x, num_iteration=reg.best_iteration))
        sub_preds_session += np.expm1(reg.predict(test_df[feats], num_iteration=reg.best_iteration)) / num_folds

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, np.sqrt(mean_squared_error(valid_y, np.log1p(oof_preds_session[valid_idx])))))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    # Full RMSEスコアの表示&LINE通知
    full_rmse_session = np.sqrt(mean_squared_error(np.log1p(train_df['totals.transactionRevenue']), np.log1p(oof_preds_session)))
    line_notify('Session Level Full RMSE score %.6f' % full_rmse_session)

    # session level feature importance
    display_importances(feature_importance_df ,
                        '../output/lgbm_importances_session.png',
                        '../output/feature_importance_lgbm_session.csv')

    ############################################################################
    # User Level predictions
    ############################################################################

    train_df['predictions'] = oof_preds_session
    test_df['predictions'] = sub_preds_session

    # Aggregate data at User level
    aggregations = {'totals.transactionRevenue': ['sum']}
    for col in feats+['totals.transactionRevenue']:
        aggregations[col] = ['sum', 'max', 'min', 'mean', 'median', 'std']

    train_df_agg = train_df[feats+['fullVisitorId','totals.transactionRevenue']].groupby('fullVisitorId').agg(aggregations)
    test_df_agg = test_df[feats + ['fullVisitorId','totals.transactionRevenue']].groupby('fullVisitorId').agg(aggregations)

    train_df_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in train_df_agg.columns.tolist()])
    test_df_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in test_df_agg.columns.tolist()])

    del oof_preds_session, sub_preds_session, train_df, test_df
    gc.collect()

    # Cross validation model
    folds_agg = get_folds(df=train_df_agg[['totals.pageviews_MEAN']].reset_index(), n_splits=num_folds)

    # Create arrays and dataframes to store results
    oof_preds_agg = np.zeros(train_df_agg.shape[0])
    sub_preds_agg = np.zeros(test_df_agg.shape[0])
    feature_importance_df_agg = pd.DataFrame()
    feats_agg = [f for f in train_df_agg.columns if f not in EXCLUDED_FEATURES+['totals.transactionRevenue_SUM']]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds_agg):
        train_x, train_y = train_df_agg[feats_agg].iloc[train_idx], np.log1p(train_df_agg['totals.transactionRevenue_SUM'].iloc[train_idx])
        valid_x, valid_y = train_df_agg[feats_agg].iloc[valid_idx], np.log1p(train_df_agg['totals.transactionRevenue_SUM'].iloc[valid_idx])

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # パラメータは適当です
        params ={
                'device' : 'gpu',
#                'gpu_use_dp':True,
                'task': 'train',
                'boosting': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'num_iteration': 10000,
                'learning_rate': 0.02,
                'num_leaves': 64,
                'colsample_bytree': 0.553240348074409,
                'subsample': 0.471522873020333,
                'max_depth': 8,
                'reg_alpha': 9.83318745912308,
                'reg_lambda': 0.925142409255232,
                'min_split_gain': 0.954402595384603,
                'min_child_weight': 44,
                'min_data_in_leaf': 79,
                'verbose': -1,
                'seed':int(2**n_fold),
                'bagging_seed':int(2**n_fold),
                'drop_seed':int(2**n_fold)
                }

        reg = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        oof_preds_agg[valid_idx] = np.expm1(reg.predict(valid_x, num_iteration=reg.best_iteration))
        sub_preds_agg += np.expm1(reg.predict(test_df_agg[feats_agg], num_iteration=reg.best_iteration)) / num_folds

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats_agg
        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df_agg = pd.concat([feature_importance_df_agg, fold_importance_df], axis=0)
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, np.sqrt(mean_squared_error(valid_y, np.log1p(oof_preds_agg[valid_idx])))))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    # Full RMSEスコアの表示&LINE通知
    full_rmse_agg = np.sqrt(mean_squared_error(np.log1p(train_df_agg['totals.transactionRevenue_SUM']), np.log1p(oof_preds_agg)))
    line_notify('Visitor Level Full RMSE score %.6f' % full_rmse_agg)

    # session level feature importance
    display_importances(feature_importance_df_agg ,
                        '../output/lgbm_importances_agg.png',
                        '../output/feature_importance_lgbm_agg.csv')

    if not debug:
        # 提出データの予測値を保存
        test_df_agg.loc[:,'PredictedLogRevenue'] = sub_preds_agg
        submission = test_df_agg[['PredictedLogRevenue']]
        submission['PredictedLogRevenue'] = np.log1p(submission['PredictedLogRevenue'])
        submission['PredictedLogRevenue'] =  submission['PredictedLogRevenue'].apply(lambda x : 0.0 if x < 0 else x)
        submission['PredictedLogRevenue'] = submission['PredictedLogRevenue'].fillna(0)
        submission.to_csv(submission_file_name, index=True)

        # out of foldの予測値を保存
        train_df_agg['OOF_PRED'] = oof_preds_agg
        train_df_agg[['OOF_PRED']].to_csv(oof_file_name, index= True)

        # API経由でsubmit
        submit(submission_file_name, comment='cv: %.6f' % full_rmse_agg)

    return feature_importance_df

def main(debug = False):
    num_rows = 10000 if debug else None
    with timer("Preprocessing"):
        df = get_df(num_rows)
        print("df shape:", df.shape)
    with timer("Run LightGBM with kfold"):
        kfold_lightgbm(df, num_folds=NUM_FOLDS, stratified=False, debug=debug)

if __name__ == "__main__":
    submission_file_name = "../output/submission.csv"
    oof_file_name = "../output/oof_lgbm.csv"
    with timer("Full model run"):
        main(debug = False)
