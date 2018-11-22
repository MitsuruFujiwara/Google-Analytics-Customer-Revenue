
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
from Utils import loadpkl, save2pkl, line_notify, submit
from Utils import EXCLUDED_FEATURES, NUM_FOLDS

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
def kfold_lightgbm(df, num_folds, stratified = False, debug= False, use_pkl=False):

    # Divide in training/validation and test data
    train_df = df[~df['IS_TEST']]
    test_df = df[df['IS_TEST']]

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()

    # Cross validation model
    folds = get_folds(df=train_df, n_splits=num_folds)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in EXCLUDED_FEATURES+['totals.transactionRevenue_SUM']]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds):
        train_x, train_y = train_df[feats].iloc[train_idx], np.log1p(train_df['totals.transactionRevenue_SUM'].iloc[train_idx])
        valid_x, valid_y = train_df[feats].iloc[valid_idx], np.log1p(train_df['totals.transactionRevenue_SUM'].iloc[valid_idx])

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # params estimated by bayesian optimization
        params ={
                'device' : 'gpu',
#                'gpu_use_dp':True,
                'task': 'train',
                'boosting': 'gbdt',
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.01,
                'num_leaves': 64,
                'colsample_bytree': 0.423362771564478,
                'subsample': 0.28631547518949,
                'max_depth': 7,
                'reg_alpha': 7.00011917374063,
                'reg_lambda': 1.57317923913574,
                'min_split_gain': 0.59000348557552,
                'min_child_weight': 41,
                'min_data_in_leaf': 2,
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
                        num_boost_round=10000,
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        oof_preds[valid_idx] = np.expm1(reg.predict(valid_x, num_iteration=reg.best_iteration))
        sub_preds += np.expm1(reg.predict(test_df[feats], num_iteration=reg.best_iteration)) / num_folds

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, np.sqrt(mean_squared_error(valid_y, np.log1p(oof_preds[valid_idx])))))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    # Full RMSEスコアの表示&LINE通知
    full_rmse = np.sqrt(mean_squared_error(np.log1p(train_df['totals.transactionRevenue_SUM']), np.log1p(oof_preds)))
    line_notify('Visitor Level Full RMSE score %.6f' % full_rmse)

    # session level feature importance
    display_importances(feature_importance_df,
                        '../output/lgbm_importances.png',
                        '../output/feature_importance_lgbm.csv')

    if not debug:
        # 提出データの予測値を保存
        test_df.loc[:,'PredictedLogRevenue'] = sub_preds
        submission = test_df[['fullVisitorId', 'PredictedLogRevenue']]
        submission.loc[:,'PredictedLogRevenue'] = np.log1p(submission['PredictedLogRevenue'])
        submission.loc[:,'PredictedLogRevenue'] =  submission['PredictedLogRevenue'].apply(lambda x : 0.0 if x < 0 else x)
        submission.loc[:,'PredictedLogRevenue'] = submission['PredictedLogRevenue'].fillna(0)
        submission.to_csv(submission_file_name, index=False)

        # out of foldの予測値を保存
        train_df.loc[:,'OOF_PRED'] = oof_preds
        train_df[['fullVisitorId', 'OOF_PRED']].to_csv(oof_file_name, index=False)

        # API経由でsubmit
        submit(submission_file_name, comment='cv: %.6f' % full_rmse)

    return feature_importance_df

def main(debug=False, use_pkl=False):
    num_rows = 10000 if debug else None
    with timer("Preprocessing"):
        df = get_df(num_rows) if not use_pkl else loadpkl('../output/df.pkl')
#        save2pkl('../output/df.pkl', df)
        print("df shape:", df.shape)
    with timer("Run LightGBM with kfold"):
        kfold_lightgbm(df, num_folds=NUM_FOLDS, stratified=False, debug=debug, use_pkl=use_pkl)

if __name__ == "__main__":
    submission_file_name = "../output/submission.csv"
    oof_file_name = "../output/oof_lgbm.csv"
    with timer("Full model run"):
        main(debug=False, use_pkl=False)
