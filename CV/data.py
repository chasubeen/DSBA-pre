import os
import gdown
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from config import cfg



### 데이터 준비
## 데이터 저장 경로 확인
data_dir = cfg.data.data_dir
os.makedirs(data_dir, exist_ok = True)  

## 데이터 다운로드
if not all(os.path.exists(os.path.join(data_dir, file)) for file in ["train_data.npy", "train_target.npy", "test_data.npy", "test_target.npy"]):
    print(f"{data_dir}에 데이터를 저장합니다.")
    gdown.download_folder(cfg.data.download_url, quiet=False, use_cookies=False)
else:
    print("\n 데이터가 이미 존재하므로 다운로드를 건너뜁니다.")



### 데이터 로드
## 데이터 경로 설정
train_data_path = os.path.join(data_dir, "train_data.npy")
train_target_path = os.path.join(data_dir, "train_target.npy")
test_data_path = os.path.join(data_dir, "test_data.npy")
test_target_path = os.path.join(data_dir, "test_target.npy")


## 데이터 존재여부 확인
if not all(os.path.exists(path) for path in [train_data_path, train_target_path, test_data_path, test_target_path]):
    raise FileNotFoundError("❌ 1개 이상의 데이터 파일이 존재하지 않습니다. 데이터 다운로드를 확인하세요.")


## 데이터 파일(.npy) 불러오기
train_data = np.load(train_data_path)
train_target = np.load(train_target_path)
test_data = np.load(test_data_path)
test_target = np.load(test_target_path)

print("\n 데이터가 정상적으로 load되었습니다.")



### DataSet & DataLoader
## configs
batch_size = cfg.data.batch_size  # 512
num_workers = cfg.data.num_workers  # 4


## 전처리
image_size = cfg.data.image_size
normalize_mean = cfg.data.normalize.mean
normalize_std = cfg.data.normalize.std

transform = transforms.Compose([
    transforms.Resize(image_size),  
    transforms.ToTensor(),
    transforms.Normalize(mean = normalize_mean, std = normalize_std)
])


## Dataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data  
        self.targets = torch.tensor(targets, dtype = torch.long)
        self.model_type = None  

    def set_model_type(self, model_type):
        # 모델 타입을 동적으로 설정(ResNet 또는 ViT)
        self.model_type = model_type

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if self.model_type is None:
            raise ValueError("모델 타입이 설정되지 않았습니다. (set_model_type()을 먼저 호출하세요.)")

        img = self.data[idx]
        label = self.targets[idx]

        # PIL 변환 후 공통 전처리 적용
        img = img.astype(np.uint8)
        img = transforms.ToPILImage()(img)
        img = transform(img)  

        if self.model_type == "resnet":
            return img, label
        elif self.model_type == "vit":
            return {"pixel_values": img, "labels": label}


## Dataset 구성 
train_dataset = CustomDataset(train_data, train_target)
test_dataset = CustomDataset(test_data, test_target)

## DataLoader 구성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print("\n DataLoader 생성이 완료되었습니다.")