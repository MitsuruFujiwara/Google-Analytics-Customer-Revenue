
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
import gc

from Utils import line_notify, submit, rmse

################################################################################
# 複数モデルのoutputをブレンドして最終的なsubmitファイルを生成するスクリプト。
################################################################################

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
    main()
