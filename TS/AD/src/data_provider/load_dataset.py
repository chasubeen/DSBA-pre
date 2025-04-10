import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.timefeatures import time_features_from_date


def load_dataset(datadir, dataname, val_split_rate=0.1, time_embedding=[False, 'd']):
    base_path = os.path.join(datadir, dataname)

    ## csv 불러오기
    trn_df = pd.read_csv(os.path.join(base_path, "train.csv"))
    tst_df = pd.read_csv(os.path.join(base_path, "test.csv"))
    label_df = pd.read_csv(os.path.join(base_path, "test_label.csv"))


    ## Train/Val Split
    # print(f"[DEBUG] Raw train total: {len(trn_df)}")
    # print(f"[DEBUG] Raw train NaN: {trn_df.isna().sum().sum()}")

    # print(f"[DEBUG] Raw test total: {len(tst_df)}")
    # print(f"[DEBUG] Raw test NaN: {tst_df.isna().sum().sum()}")

    # print(f"[DEBUG] Raw test total: {len(tst_df)}")

    ## Drop NaNs from train
    trn_df = trn_df.dropna().reset_index(drop=True)

    train_df, val_df = train_test_split(trn_df, test_size=val_split_rate, shuffle=False)
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    tst_df = tst_df.reset_index(drop=True)
    

    ## Timestamp & Value 분리
    def split(df):
        ts = df.iloc[:, 0]
        x = df.iloc[:, 1:].astype(np.float32)
        return ts, x

    trn_ts, trn = split(train_df)
    val_ts, val = split(val_df)
    tst_ts, tst = split(tst_df)

    label = label_df.iloc[:, 1:].values.squeeze().astype(np.int64)

    var = trn.shape[1]  # feature 수
    
    return trn, trn_ts, val, val_ts, tst, tst_ts, var, label

