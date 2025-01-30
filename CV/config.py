from omegaconf import OmegaConf
import torch


cfg = OmegaConf.create({
    ## 데이터 설정
    "data": {
        "download_url": "https://drive.google.com/drive/folders/1wEnwMeJoQZwJhI7oBRah3I8QX-FpLMYD",
        "data_dir": "./data",
        "batch_size": 512,
        "num_workers": 4
    },

    ## 모델 설정
    "model": {
        "save_dir": "./models",  # 학습된 모델 저장 경로
        "model_save_format": "model_{model}_{pretrained}.pth"  # 모델 저장 파일 형식
    },

    ## 학습 설정
    "training": {
        "epochs": 10,
        "learning_rate": 0.001,
        "loss_function": "CrossEntropyLoss",  # 손실 함수
        "optimizer": "Adam",  # 옵티마이저
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42  # 결과 재현성 설정
    },

    ## 실험 설정
    "experiment": {
        "results_dir": "./results",  # 실험 결과 저장 경로
        "log_file": "experiment_results.txt",  # 로그 저장 파일
        "save_model": True,  # 모델 저장 여부
        "experiments": [  # 수행할 실험 목록
            {"model": "ResNet50", "pretrained": False},
            {"model": "ViT-S/16", "pretrained": False},
            {"model": "ResNet50", "pretrained": True},
            {"model": "ViT-S/16", "pretrained": True}
        ]
    }
})