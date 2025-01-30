from omegaconf import OmegaConf
import torch


cfg = OmegaConf.create({
    ## 데이터 설정
    "data": {
        "download_url": "https://drive.google.com/drive/folders/1wEnwMeJoQZwJhI7oBRah3I8QX-FpLMYD", # 원본 저장소 경로
        "data_dir": "./data", # 로컬 저장소 경로
        "batch_size": 512,
        "num_workers": 4
    },
    
    ## 모델 설정
    "model": {
        "save_dir": "./models",  # 학습된 모델을 저장할 경로
        "resnet50": {
            "pretrained": True,  # Pre-trained weight 적용 여부
            "scratch": True      # Scratch로 학습 여부
        },
        "vit": {
            "pretrained": True,
            "scratch": True
        }
    },

    ## 학습 설정
    "training": {
        "epochs": 10,               # 학습 epoch 수
        "learning_rate": 0.001,     # 학습률
        "loss_function": "CrossEntropyLoss",  # Loss 함수
        "optimizer": "Adam",        # Optimizer 종류
        "device": "cuda" if torch.cuda.is_available() else "cpu",  # Device 설정
        "seed": 42                  # Seed 값 (결과 재현성)
    },
    
    ## 실험 설정
    "experiment": {
        "results_dir": "./results",  # 결과 저장 경로
        "save_model": True           # 모델 저장 여부
    }
})