from omegaconf import OmegaConf
import torch


cfg = OmegaConf.create({
    ## 데이터 설정
    "data": {
        "download_url": "https://drive.google.com/drive/folders/1wEnwMeJoQZwJhI7oBRah3I8QX-FpLMYD",
        "data_dir": "./data",
        "batch_size": 64,
        "num_workers": 4,
        "image_size": (224, 224),  
        "normalize": {  
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5]
        }
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
        "results_dir": "./results",
        "log_file": "experiment_results.txt",
        "save_model": True,
        "experiments": [
            {"model": "ResNet50", "model_type": "resnet", "pretrained": False},
            {"model": "ViT-S/16", "model_type": "vit", "pretrained": False},
            {"model": "ResNet50", "model_type": "resnet", "pretrained": True},
            {"model": "ViT-S/16", "model_type": "vit", "pretrained": True}
        ]
    }
})