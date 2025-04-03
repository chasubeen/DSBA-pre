from torch.utils.data import Dataset

from pathlib import Path
from datetime import timedelta
from utils.timefeatures import time_features
import dateutil
import pdb
from omegaconf import OmegaConf

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features_from_date


def load_dataset(
    datadir: str,
    dataname: str,
    split_rate: list,
    time_embedding: list = [True, 'h'],
    del_feature: list = None
):
    filepath = Path(datadir) / f"{dataname}.csv"
    df = pd.read_csv(filepath)
    
    df.columns = [col.lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    if del_feature:
        df.drop(columns=del_feature, inplace=True)

    ts = df['date'].copy()
    df_data = df.drop(columns=['date'])

    if time_embedding[0]:
        time_feat = time_features_from_date(ts, time_embedding[1])
        df_data = pd.concat([df_data, time_feat], axis=1)

    n = len(df_data)
    trn_end = int(n * split_rate[0])
    val_end = int(n * (split_rate[0] + split_rate[1]))

    trn = df_data[:trn_end].values
    val = df_data[trn_end:val_end].values
    tst = df_data[val_end:].values

    trn_ts = ts[:trn_end].values
    val_ts = ts[trn_end:val_end].values
    tst_ts = ts[val_end:].values

    var = df_data.shape[1]

    return trn, trn_ts, val, val_ts, tst, tst_ts, var