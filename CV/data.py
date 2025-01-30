import os
import gdown
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import cfg


## 데이터 다운로드
# 데이터 저장 경로 확인
data_dir = cfg.data.data_dir
os.makedirs(data_dir, exist_ok = True)  

print(f"{data_dir}에 데이터를 저장합니다.")
gdown.download_folder(cfg.data.download_url, quiet = False, use_cookies = False)


## 데이터 로드
# 데이터 경로 설정
train_data_path = os.path.join(data_dir, "train_data.npy")
train_target_path = os.path.join(data_dir, "train_target.npy")
test_data_path = os.path.join(data_dir, "test_data.npy")
test_target_path = os.path.join(data_dir, "test_target.npy")

if not all(os.path.exists(path) for path in [train_data_path, train_target_path, test_data_path, test_target_path]):
    raise FileNotFoundError("1개 이상의 파일이 존재하지 않습니다.")

# 데이터 파일(.npy) 불러오기
train_data = np.load(train_data_path)
train_target = np.load(train_target_path)
test_data = np.load(test_data_path)
test_target = np.load(test_target_path)

print("\n 데이터가 정상적으로 load되었습니다.")


## DataLoader, Dataset 생성
# configs
batch_size = cfg.data.batch_size # 32
num_workers = cfg.data.num_workers # 4

# tensor로 변경
# pytorch는 [C,H,W] 순서가 표준 -> [N, C, H, W] 형태로
train_data_tensor = torch.tensor(train_data, dtype = torch.float32).permute(0, 3, 1, 2) 
train_target_tensor = torch.tensor(train_target, dtype = torch.long)
test_data_tensor = torch.tensor(test_data, dtype = torch.float32).permute(0, 3, 1, 2)
test_target_tensor = torch.tensor(test_target, dtype = torch.long)

# Dataset 구성
train_dataset = TensorDataset(train_data_tensor, train_target_tensor)
test_dataset = TensorDataset(test_data_tensor, test_target_tensor)

# DataLoader 구성
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

print("\n DataLoader 생성이 완료되었습니다.")