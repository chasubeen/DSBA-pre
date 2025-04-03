from torch.utils.data import Dataset
import numpy as np


class BuildDataset(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len, target_index=None):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len # context 길이
        self.pred_len = pred_len
        self.target_index = target_index

        # 생성 가능한 데이터 샘플 수
        self.valid_window = len(data) - seq_len - pred_len + 1
    
    def __getitem__(self, idx):
        start = idx
        end_input = start + self.seq_len
        end_label = end_input + self.pred_len

        past_window = self.data[start:end_input] 
        future_window = self.data[end_input:end_label]

        if self.target_index is not None:
            future_window = future_window[:, self.target_index]

        # x(past_window): [batch_size, seq_len, num_features]
        # y(future_window): [batch_size, pred_len, num_features]
        return past_window, future_window
    
    def __len__(self):        
        return self.valid_window