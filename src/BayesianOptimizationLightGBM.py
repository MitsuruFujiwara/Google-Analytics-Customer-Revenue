import gc
import numpy as np
import pandas as pd
import lightgbm

from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, StratifiedKFold

from Preprocessing_326 import get_df
from Learning_lgbm_326 import get_folds
from Utils import NUM_FOLDS, EXCLUDED_FEATURES

# 以下参考
# https://github.com/fmfn/BayesianOptimization
# https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code

NUM_ROWS=None

DF = get_df(NUM_ROWS)

# split test & train
TRAIN_DF = DF[~DF['IS_TEST']]
FEATS = [f for f in TRAIN_DF.columns if f not in EXCLUDED_FEATURES+['totals.transactionRevenue']]

lgbm_train = lightgbm.Dataset(TRAIN_DF[FEATS],
                              np.log1p(TRAIN_DF['totals.transactionRevenue']),
                              free_raw_data=False
                              )

def lgbm_eval(num_leaves,
              colsample_bytree,
              subsample,
              max_depth,
              reg_alpha,
              reg_lambda,
              min_split_gain,
              min_child_weight,
              min_data_in_leaf
              ):

    params = dict()
    params["learning_rate"] = 0.02
#    params["silent"] = True
    params['device'] = 'gpu'
#    params["nthread"] = 16
    params['objective'] = 'regression'
    params['seed']=326,

    params["num_leaves"] = int(num_leaves)
    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['subsample'] = max(min(subsample, 1), 0)
    params['max_depth'] = int(max_depth)
    params['reg_alpha'] = max(reg_alpha, 0)
    params['reg_lambda'] = max(reg_lambda, 0)
    params['min_split_gain'] = min_split_gain
    params['min_child_weight'] = min_child_weight
    params['min_data_in_leaf'] = int(min_data_in_leaf)
    params['verbose']=-1

    folds = get_folds(df=TRAIN_DF, n_splits=NUM_FOLDS)

    clf = lightgbm.cv(params=params,
                      train_set=lgbm_train,
                      metrics=['rmse'],
                      nfold=NUM_FOLDS,
                      folds=folds,
                      num_boost_round=10000, # early stopありなのでここは大きめの数字にしてます
                      early_stopping_rounds=200,
                      verbose_eval=100,
                      seed=47,
                     )
    gc.collect()
    return -clf['rmse-mean'][-1]

def main():

    # clf for bayesian optimization
    clf_bo = BayesianOptimization(lgbm_eval, {'num_leaves': (16, 64),
                                              'colsample_bytree': (0.001, 1),
                                              'subsample': (0.001, 1),
                                              'max_depth': (3, 8),
                                              'reg_alpha': (0, 10),
                                              'reg_lambda': (0, 10),
                                              'min_split_gain': (0, 1),
                                              'min_child_weight': (0, 45),
                                              'min_data_in_leaf': (0, 500),
                                              })

    clf_bo.maximize(init_points=15, n_iter=25)

    res = pd.DataFrame(clf_bo.res['max']['max_params'], index=['max_params'])

    res.to_csv('../output/max_params_lgbm.csv')

if __name__ == '__main__':
    main()
