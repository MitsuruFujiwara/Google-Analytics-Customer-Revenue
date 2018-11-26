import gc
import numpy as np
import pandas as pd
import xgboost

from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, StratifiedKFold

from Preprocessing_326 import get_df
from Learning_lgbm_326 import get_folds
from Utils import NUM_FOLDS, EXCLUDED_FEATURES, read_pickles, line_notify

# 以下参考
# https://github.com/fmfn/BayesianOptimization
# https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code

NUM_ROWS=None

TRAIN_DF = read_pickles('../output/train_df_agg_xgb')

# split test & train
#TRAIN_DF = DF[~DF['IS_TEST']]
FEATS = [f for f in TRAIN_DF.columns if f not in EXCLUDED_FEATURES+['totals.transactionRevenue_SUM']]

xgb_train = xgboost.DMatrix(TRAIN_DF[FEATS],
                        np.log1p(TRAIN_DF['totals.transactionRevenue_SUM'])
                        )

del TRAIN_DF
gc.collect()

def xgb_eval(gamma,
             max_depth,
             min_child_weight,
             subsample,
             colsample_bytree,
             colsample_bylevel,
             alpha,
             _lambda):

    params = {
            'objective':'gpu:reg:linear', # GPU parameter
            'booster': 'gbtree',
            'eval_metric':'rmse',
            'silent':1,
            'eta': 0.01,
            'tree_method': 'gpu_hist', # GPU parameter
            'predictor': 'gpu_predictor', # GPU parameter
            'seed':326
            }

    params['gamma'] = gamma
    params['max_depth'] = int(max_depth)
    params['min_child_weight'] = min_child_weight
    params['subsample'] = max(min(subsample, 1), 0)
    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['colsample_bylevel'] = max(min(colsample_bylevel, 1), 0)
    params['alpha'] = max(alpha, 0)
    params['lambda'] = max(_lambda, 0)

    reg = xgboost.cv(params=params,
                     dtrain=xgb_train,
                     num_boost_round=10000, # early stopありなのでここは大きめの数字にしてます
                     nfold=5,
                     metrics=["rmse"],
                     folds=None,
                     early_stopping_rounds=200,
                     verbose_eval=100,
                     seed=47,
                     )
    gc.collect()
    return -reg['test-rmse-mean'].iloc[-1]

def main():

    # reg for bayesian optimization
    reg_bo = BayesianOptimization(xgb_eval, {'gamma':(0, 1),
                                             'max_depth': (6, 6),
                                             'min_child_weight': (0, 45),
                                             'subsample': (0.001, 1),
                                             'colsample_bytree': (0.001, 1),
                                             'colsample_bylevel': (0.001, 1),
                                             'alpha': (9, 20),
                                             '_lambda': (0, 10)
                                             })

    reg_bo.maximize(init_points=15, n_iter=25)

    res = pd.DataFrame(reg_bo.res['max']['max_params'], index=['max_params'])

    res.to_csv('../output/max_params_xgb_user.csv')

    line_notify('xgb user finished.')

if __name__ == '__main__':
    main()
