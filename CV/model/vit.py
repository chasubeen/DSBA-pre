import torch
import timm
from torch import nn


### ViT Model
class ViT_S16(nn.Module):
    def __init__(self, num_classes = 10, pretrained = True):
        super().__init__()

        ## timm에서 ViT-S/16 불러오기
        self.model = timm.create_model(
            "vit_small_patch16_224.augreg_in1k", 
            pretrained = pretrained, 
            num_classes = num_classes
        )

        # Feature vector를 반환할 수 있도록 설정
        self.feature_dim = self.model.head.in_features  # FC layer 입력 차원 저장
        # 기존 FC layer 제거 
        self.model.head = nn.Identity()  
        # 새로운 FC layer 추가(fine-tuning 목적)
        self.fc = nn.Linear(self.feature_dim, num_classes)


    def forward(self, x):
        h = self.model(x)  # Feature Vector 추출
        x = self.fc(h)  # 최종 Classification 수행
        return x