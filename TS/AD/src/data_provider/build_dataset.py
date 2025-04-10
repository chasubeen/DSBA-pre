import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class BuildDataset(Dataset):
    def __init__(self, data, seq_len, stride_len, 
                 timestamps=None, labels=None):
        self.data = np.array(data, dtype=np.float32)
        self.seq_len = seq_len
        self.stride_len = stride_len
        self.timestamps = np.array(timestamps) if timestamps is not None else None
        self.labels = np.array(labels, dtype=np.int64) if labels is not None else None

        self.windows = self._create_windows()

        # # 디버깅용
        # print(f"[DEBUG] Dataset shape: {self.data.shape}")
        # print(f"[DEBUG] NaN in whole data: {np.isnan(self.data).any()}")

    def _create_windows(self):
        windows = []
        for i in range(0, len(self.data) - self.seq_len + 1, self.stride_len):
            x = self.data[i:i+self.seq_len]
            if np.isnan(x).any():
                continue  # skip window with NaN
            windows.append(i)
        return windows

    def __len__(self):
        return len(self.windows)

    ## 슬라이딩 윈도우 단위로 데이터 반환
    def __getitem__(self, idx):
        start = self.windows[idx]
        end = start + self.seq_len

        x = torch.tensor(self.data[start:end], dtype=torch.float32)
        item = {"input": x}

        if self.timestamps is not None:
            ts = torch.tensor(np.array(self.timestamps[start:end]), dtype=torch.long)
            item["timestamp"] = ts

        if self.labels is not None:
            y = torch.tensor(self.labels[start:end], dtype=torch.long)
            item["label"] = y

        return item