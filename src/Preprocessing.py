import pandas as pd
import gc

from Utils import load_df

def get_df(num_rows=None):
    # load datasets
    train_df = load_df('../input/train.csv', nrows=num_rows)
    test_df = load_df('../input/test.csv', nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))

    # train testの識別用
    train_df['IS_TEST'] = False
    test_df['IS_TEST'] = True

    # Merge
    df = train_df.append(test_df).reset_index()

    del train_df, test_df
    gc.collect()

    # TODO: ここから下にFeature Engineeringの処理追加していく感じで

    return df

if __name__ == '__main__':
    # test
    df = get_df(num_rows=10000)
    print(df)
