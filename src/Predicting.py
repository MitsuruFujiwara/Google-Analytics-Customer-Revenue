
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
import gc

from Utils import line_notify, NUM_FOLDS

################################################################################
# 複数モデルのoutputをブレンドして最終的なsubmitファイルを生成するスクリプト。
################################################################################

def main(debug = False):
    num_rows = 10000 if debug else None
    # load submission file
    sub = pd.read_csv('../input/index_master.csv')
    sub['高低左'] = 0.0

    for route in ['A', 'B', 'C', 'D']:
        # load df for each route
        df = train_test(route, num_rows)

        # Divide in training/validation and test data
        test_df = df[df['高低左'].isnull()]
        print("Starting Route {} Prediction. test shape: {}".format(route, test_df.shape))

        del df
        gc.collect()

        sub_preds = np.zeros(test_df.shape[0])
        feats = [f for f in test_df.columns if f not in ['id', 'date', 'キロ程', '高低左']]
        for n_fold in range(NUM_FOLDS):
            # load model
            reg = lgb.Booster(model_file='../output/lgbm_'+route+'_'+str(n_fold)+'.txt')

            # prediction
            sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / NUM_FOLDS
            print('Route {} fold {} finished'.format(route, n_fold+1))

            del reg
            gc.collect()

        # 提出データの予測値を保存
        sub.loc[sub['路線']==route,'高低左'] = sub_preds

        # LINE通知
        line_notify("Finished Route {} Prediction.".format(route))

        del sub_preds, test_df
        gc.collect()

    sub['高低左'].to_csv(submission_file_name, header=False)

    # LINE通知
    line_notify('Saved {}'.format(submission_file_name))

if __name__ == '__main__':
    submission_file_name = "../output/submission.csv"
    main(debug=False)
