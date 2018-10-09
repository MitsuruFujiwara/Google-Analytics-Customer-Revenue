
import lightgbm as lgb
import numpy as np
import pandas as pd
import gc
import time

from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from Preprocessing import get_df
from Utils import EXCLUDED_FEATURES

################################################################################
# とりあえずHome Creditの改変です
################################################################################

warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append([ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                         ids[df['fullVisitorId'].isin(unique_vis[val_vis])]])

    return fold_ids

# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(df, n_splits, debug= False):

    # Divide in training/validation and test data
    train_df = df[~df['IS_TEST']]
    test_df = df[df['IS_TEST']]

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()

    # Cross validation model
    folds = get_folds(train_df, n_splits)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in EXCLUDED_FEATURES]

    # 最初にsplitしないバージョンでモデルを推定します
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               free_raw_data=False)

        # new params https://github.com/neptune-ml/open-solution-home-credit/wiki/LightGBM-clean-dynamic-features
        params = {
                'device' : 'gpu',
#                'gpu_use_dp':True, #これで倍精度演算できるっぽいです
                'task': 'train',
#                'boosting_type': 'dart',
                'objective': 'binary',
                'metric': {'rmse'},
                'num_threads': -1,
                'learning_rate': 0.02,
                'num_iteration': 10000,
                'num_leaves': 39,
                'colsample_bytree': 0.0587705926,
                'subsample': 0.5336340435,
                'max_depth': 7,
                'reg_alpha': 8.9675927624,
                'reg_lambda': 9.8953903428,
                'min_split_gain': 0.911786867,
                'min_child_weight': 37,
                'min_data_in_leaf': 629,
                'verbose': -1,
                'seed':int(2**n_fold),
                'bagging_seed':int(2**n_fold),
                'drop_seed':int(2**n_fold)
                }

        clf = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        sub_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain', iteration=clf.best_iteration)
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

    if not debug:
        # 提出データの予測値を保存
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)

        # out of foldの予測値を保存
        train_df['OOF_PRED'] = oof_preds
        train_df[['SK_ID_CURR', 'OOF_PRED']].to_csv(oof_file_name, index= False)

    return feature_importance_df

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

def main(debug = False):
    num_rows = 10000 if debug else None
    with timer("Preprocessing"):
        df = get_df(num_rows)
        print("df shape:", df.shape)
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds=5, stratified=True, debug= debug)
        display_importances(feat_importance ,'../output/lgbm_importances.png', '../output/feature_importance_lgbm.csv')

if __name__ == "__main__":
    submission_file_name = "../output/submission.csv"
    oof_file_name = "../output/oof_lgbm.csv"
    with timer("Full model run"):
        main(debug = False)
