
import xgboost as xgb
import numpy as np
import pandas as pd
import gc
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from contextlib import contextmanager
from sklearn.model_selection import GroupKFold
from pandas.core.common import SettingWithCopyWarning

from Preprocessing_326 import get_df
from Utils import line_notify, submit, rmse, to_pickles, read_pickles
from Utils import EXCLUDED_FEATURES, NUM_FOLDS

################################################################################
# Kuso-simple XGBoost k-fold
################################################################################

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

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
    plt.title('XGBoost Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outputpath)

# XGBoost with KFold or Stratified KFold
def kfold_xgboost(df, num_folds, stratified = False, debug= False, use_pkl=False):

    # Divide in training/validation and test data
    train_df = df[~df['IS_TEST']]
    test_df = df[df['IS_TEST']]

    print("Starting XGBoost. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()

    ############################################################################
    # Session Level predictions
    ############################################################################

    print('Starting Session Level predictions...')

    # Cross validation model
    folds_session = get_folds(df=train_df, n_splits=num_folds)

    # Create arrays and dataframes to store results
    oof_preds_session = np.zeros(train_df.shape[0])
    sub_preds_session = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in EXCLUDED_FEATURES+['totals.transactionRevenue']]

    # final predict用にdmatrix形式のtest dfを作っておきます
    test_df_dmtrx = xgb.DMatrix(test_df[feats], label=train_df['totals.transactionRevenue'])

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds_session):
        train_x, train_y = train_df[feats].iloc[train_idx], np.log1p(train_df['totals.transactionRevenue'].iloc[train_idx])
        valid_x, valid_y = train_df[feats].iloc[valid_idx], np.log1p(train_df['totals.transactionRevenue'].iloc[valid_idx])

        # set data structure
        xgb_train = xgb.DMatrix(train_x,
                                label=train_y)
        xgb_test = xgb.DMatrix(valid_x,
                               label=valid_y)

        # params
        params = {
                'objective':'gpu:reg:linear', # GPU parameter
                'booster': 'gbtree',
                'eval_metric':'rmse',
                'silent':1,
                'eta': 0.01,
                'max_depth': 8,
                'min_child_weight': 44,
                'gamma': 0.5997057606,
                'subsample': 0.6221326906,
                'colsample_bytree': 0.6405879054,
                'colsample_bylevel': 0.9772125093,
                'alpha':9.83318745912308,
                'lambda': 0.925142409255232,
                'tree_method': 'gpu_hist', # GPU parameter
                'predictor': 'gpu_predictor', # GPU parameter
                'seed':int(2**n_fold)
                }

        reg = xgb.train(
                        params,
                        xgb_train,
                        num_boost_round=10000,
                        evals=[(xgb_train,'train'),(xgb_test,'test')],
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        oof_preds_session[valid_idx] = np.expm1(reg.predict(xgb_test))
        sub_preds_session += np.expm1(reg.predict(test_df_dmtrx)) / num_folds

        fold_importance_df = pd.DataFrame.from_dict(reg.get_score(importance_type='gain'), orient='index', columns=['importance'])
        fold_importance_df["feature"] = fold_importance_df.index.tolist()
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, np.log1p(oof_preds_session[valid_idx]))))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    del test_df_dmtrx
    gc.collect()

    # Full RMSEスコアの表示&LINE通知
    full_rmse_session = rmse(np.log1p(train_df['totals.transactionRevenue']), np.log1p(oof_preds_session))
    line_notify('XGBoost Session Level Full RMSE score %.6f' % full_rmse_session)

    # session level feature importance
    display_importances(feature_importance_df ,
                        '../output/xgb_importances_session.png',
                        '../output/feature_importance_xgb_session.csv')

    # 予測値を保存
    train_df.loc[:,'predictions'] = oof_preds_session
    test_df.loc[:,'predictions'] = sub_preds_session

    del oof_preds_session, sub_preds_session
    gc.collect()

    # csv形式でsave
    train_df['predictions'].to_csv("../output/oof_xgb_session.csv")
    test_df['predictions'].to_csv("../output/sub_xgb_session.csv")

    ############################################################################
    # User Level predictions
    ############################################################################

    print('Starting User Level predictions...')

    if use_pkl:
        # load pkl
        train_df_agg = read_pickles('../output/train_df_agg_xgb')
        test_df_agg = read_pickles('../output/test_df_agg_xgb')
    else:
        # Aggregate data at User level
        aggregations = {'totals.transactionRevenue': ['sum']}
        for col in feats+['predictions']:
            aggregations[col] = ['sum', 'max', 'min', 'mean', 'std']

        train_df_agg = train_df[feats+['fullVisitorId','totals.transactionRevenue', 'predictions']].groupby('fullVisitorId').agg(aggregations)
        train_df_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in train_df_agg.columns.tolist()])
        train_df_agg = train_df_agg.astype('float32')

        # Create a list of predictions for each Visitor
        with timer("Create a list of predictions for each Visitor"):
            trn_pred_list = train_df[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
                .apply(lambda df: list(df.predictions))\
                .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})

        del train_df
        gc.collect()

        # Create a DataFrame with VisitorId as index
        # trn_pred_list contains dict
        # so creating a dataframe from it will expand dict values into columns
        trn_all_predictions = pd.DataFrame(list(trn_pred_list.values), index=train_df_agg.index)
        trn_feats = trn_all_predictions.columns
        trn_all_predictions.loc[:,'t_mean'] = np.log1p(trn_all_predictions[trn_feats].mean(axis=1)).astype('float32')
        trn_all_predictions.loc[:,'t_median'] = np.log1p(trn_all_predictions[trn_feats].median(axis=1)).astype('float32')
        trn_all_predictions.loc[:,'t_sum_log'] = np.log1p(trn_all_predictions[trn_feats]).sum(axis=1).astype('float32')
        trn_all_predictions.loc[:,'t_sum_act'] = np.log1p(trn_all_predictions[trn_feats].fillna(0).sum(axis=1)).astype('float32')
        trn_all_predictions.loc[:,'t_nb_sess'] = trn_all_predictions[trn_feats].isnull().sum(axis=1).astype('float32')
        train_df_agg = pd.concat([train_df_agg, trn_all_predictions], axis=1)

        # save pkl
        to_pickles(train_df_agg, '../output/train_df_agg_xgb', split_size=50, inplace=False)

        del trn_all_predictions, trn_pred_list
        gc.collect()

        test_df_agg = test_df[feats + ['fullVisitorId','totals.transactionRevenue', 'predictions']].groupby('fullVisitorId').agg(aggregations)
        test_df_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in test_df_agg.columns.tolist()])
        test_df_agg = test_df_agg.astype('float32')

        # Create a list of predictions for each Visitor
        with timer("Create a list of predictions for each Visitor"):
            sub_pred_list = test_df[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
                .apply(lambda df: list(df.predictions))\
                .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})

        del test_df
        gc.collect()

        sub_all_predictions = pd.DataFrame(list(sub_pred_list.values), index=test_df_agg.index)
        for f in trn_feats:
            if f not in sub_all_predictions.columns:
                sub_all_predictions[f] = np.nan
        sub_all_predictions.loc[:,'t_mean'] = np.log1p(sub_all_predictions[trn_feats].mean(axis=1)).astype('float32')
        sub_all_predictions.loc[:,'t_median'] = np.log1p(sub_all_predictions[trn_feats].median(axis=1)).astype('float32')
        sub_all_predictions.loc[:,'t_sum_log'] = np.log1p(sub_all_predictions[trn_feats]).sum(axis=1).astype('float32')
        sub_all_predictions.loc[:,'t_sum_act'] = np.log1p(sub_all_predictions[trn_feats].fillna(0).sum(axis=1)).astype('float32')
        sub_all_predictions.loc[:,'t_nb_sess'] = sub_all_predictions[trn_feats].isnull().sum(axis=1).astype('float32')
        test_df_agg = pd.concat([test_df_agg, sub_all_predictions], axis=1)

        del sub_all_predictions, sub_pred_list, trn_feats
        gc.collect()

        # save pkl
        to_pickles(test_df_agg, '../output/test_df_agg_xgb', split_size=5, inplace=False)

    # Cross validation model
    folds_agg = get_folds(df=train_df_agg[['totals.pageviews_MEAN']].reset_index(), n_splits=num_folds)

    # Create arrays and dataframes to store results
    oof_preds_agg = np.zeros(train_df_agg.shape[0])
    sub_preds_agg = np.zeros(test_df_agg.shape[0])
    feature_importance_df_agg = pd.DataFrame()
    feats_agg = [f for f in train_df_agg.columns if f not in EXCLUDED_FEATURES+['totals.transactionRevenue_SUM']]

    # final predict用にdmatrix形式のtest dfを作っておきます
    test_df_dmtrx_agg = xgb.DMatrix(test_df_agg[feats_agg], label=train_df_agg['totals.transactionRevenue_SUM'])

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds_agg):
        train_x, train_y = train_df_agg[feats_agg].iloc[train_idx], np.log1p(train_df_agg['totals.transactionRevenue_SUM'].iloc[train_idx])
        valid_x, valid_y = train_df_agg[feats_agg].iloc[valid_idx], np.log1p(train_df_agg['totals.transactionRevenue_SUM'].iloc[valid_idx])

        # set data structure
        xgb_train = xgb.DMatrix(train_x,
                                label=train_y)
        xgb_test = xgb.DMatrix(valid_x,
                               label=valid_y)

        # params
        params = {
                'objective':'gpu:reg:linear', # GPU parameter
                'booster': 'gbtree',
                'eval_metric':'rmse',
                'silent':1,
                'eta': 0.01,
                'max_depth': 8,
                'min_child_weight': 44,
                'gamma': 0.5997057606,
                'subsample': 0.6221326906,
                'colsample_bytree': 0.6405879054,
                'colsample_bylevel': 0.9772125093,
                'alpha':9.83318745912308,
                'lambda': 0.925142409255232,
                'tree_method': 'gpu_hist', # GPU parameter
                'predictor': 'gpu_predictor', # GPU parameter
                'seed':int(2**n_fold)
                }

        reg = xgb.train(
                        params,
                        xgb_train,
                        num_boost_round=10000,
                        evals=[(xgb_train,'train'),(xgb_test,'test')],
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        oof_preds_agg[valid_idx] = np.expm1(reg.predict(xgb_test))
        sub_preds_agg += np.expm1(reg.predict(test_df_dmtrx_agg)) / num_folds

        fold_importance_df = pd.DataFrame.from_dict(reg.get_score(importance_type='gain'), orient='index', columns=['importance'])
        fold_importance_df["feature"] = fold_importance_df.index.tolist()
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df_agg = pd.concat([feature_importance_df_agg, fold_importance_df], axis=0)

        print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, np.log1p(oof_preds_agg[valid_idx]))))
        del reg, train_x, train_y, valid_x, valid_y, fold_importance_df
        gc.collect()

    # Full RMSEスコアの表示&LINE通知
    full_rmse_agg = rmse(np.log1p(train_df_agg['totals.transactionRevenue_SUM']), np.log1p(oof_preds_agg))
    line_notify('Visitor Level Full RMSE score %.6f' % full_rmse_agg)

    # session level feature importance
    display_importances(feature_importance_df_agg ,
                        '../output/xgb_importances_agg.png',
                        '../output/feature_importance_xgb_agg.csv')

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

def main(debug=False, use_pkl=False):
    num_rows = 10000 if debug else None
    with timer("Preprocessing"):
        df = get_df(num_rows) if not use_pkl else read_pickles('../output/df')
        to_pickles(df, '../output/df', split_size=30)
        print("df shape:", df.shape)
    with timer("Run XGBoost with kfold"):
        kfold_xgboost(df, num_folds=NUM_FOLDS, stratified=False, debug=debug, use_pkl=use_pkl)

if __name__ == "__main__":
    submission_file_name = "../output/submission_xgb.csv"
    oof_file_name = "../output/oof_xgb.csv"
    with timer("Full model run"):
        main(debug=False, use_pkl=True)
