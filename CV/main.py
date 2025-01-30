import torch
import torch.optim as optim
import torch.nn as nn
from resnet50 import ResNet50
from train_eval import trainer, evaluator
from data import train_loader, test_loader
import timm
from config import cfg
from metrics import accuracy, top_k_error, precision, recall, f1


### 실험 설정
device = cfg.training.device
experiments = cfg.experiment.experiments
log_file = cfg.experiment.log_file
loss_function = getattr(nn, cfg.training.loss_function)
optimizer_class = getattr(optim, cfg.training.optimizer)

### 실험 수행
with open(log_file, "w") as f:
    for exp in experiments:
        model_name, pretrained = exp["model"], exp["pretrained"]

        print(f"\n▶ Running Experiment: {model_name} (Pretrained: {pretrained})")
        f.write(f"\n▶ Running Experiment: {model_name} (Pretrained: {pretrained})\n")

        ## 모델 선택
        if model_name == "ResNet50":
            model = ResNet50(num_classes=10).to(device)
            if pretrained:
                model.load_state_dict(torch.hub.load_state_dict_from_url(
                    "https://download.pytorch.org/models/resnet50-19c8e357.pth",
                    map_location=device
                ), strict=False)
        elif model_name == "ViT-S/16":
            model = timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=10).to(device)

        ## 손실 함수 및 옵티마이저 초기화
        criterion = loss_function()
        optimizer = optimizer_class(model.parameters(), lr=cfg.training.learning_rate)

        ## 학습 및 평가 루프
        for epoch in range(cfg.training.epochs):
            print(f"Epoch {epoch + 1}/{cfg.training.epochs}")
            f.write(f"Epoch {epoch + 1}/{cfg.training.epochs}\n")

            # 학습
            train_loss, train_acc = trainer(model, train_loader, criterion, optimizer, device)
            log_msg = f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%"
            print(log_msg)
            f.write(log_msg + "\n")

            # 평가
            test_metrics = evaluator(model, test_loader, criterion, device)
            eval_results = {
                "Test Loss": test_metrics["loss"],
                "Test Accuracy": test_metrics["accuracy"] * 100,
                "Top-5 Error": top_k_error(test_metrics["logits"], test_metrics["targets"]) * 100,
                "Precision": precision(test_metrics["logits"], test_metrics["targets"]),
                "Recall": recall(test_metrics["logits"], test_metrics["targets"]),
                "F1 Score": f1(test_metrics["logits"], test_metrics["targets"])
            }

            log_msg = "  " + ", ".join([f"{key}: {value:.4f}" for key, value in eval_results.items()])
            print(log_msg)
            f.write(log_msg + "\n")

        ## 모델 저장
        save_path = f"{cfg.model.save_dir}/{cfg.model.model_save_format.format(model=model_name, pretrained='pretrained' if pretrained else 'scratch')}"
        torch.save(model.state_dict(), save_path)
        log_msg = f" 모델 저장 완료: {save_path}"
        print(log_msg)
        f.write(log_msg + "\n")
