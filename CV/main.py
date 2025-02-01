import os
import torch
import torch.optim as optim
import torch.nn as nn

from config import cfg
from data import train_loader, test_loader, train_dataset, test_dataset
from model.resnet50 import ResNet50
import timm
from train_eval import trainer, evaluator
# from metrics import accuracy, top_k_error, precision, recall, f1



### 실험 설정
device = cfg.training.device
experiments = cfg.experiment.experiments
log_file = cfg.experiment.log_file
loss_function = getattr(nn, cfg.training.loss_function)
optimizer_class = getattr(optim, cfg.training.optimizer)


## 로그 파일 초기화
os.makedirs(cfg.experiment.results_dir, exist_ok = True)
log_path = os.path.join(cfg.experiment.results_dir, log_file)


## learning rate scheduling
epochs = cfg.training.epochs
warmup_epochs = cfg.training.warmup_epochs
initial_lr = cfg.training.learning_rate

def get_scheduler(optimizer, total_epochs, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear warmup
        else:
            # cosine annealing Scheduler
            return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / (total_epochs - warmup_epochs) * 3.141592653589793)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



### 실험 수행
with open(log_path, "w") as f:
    for exp in experiments:
        model_name, model_type, pretrained = exp["model"], exp["model_type"], exp["pretrained"]

        print(f"\n=== Running Experiment: {model_name}(Pretrained: {pretrained}) ===")
        f.write(f"\n=== Running Experiment: {model_name}(Pretrained: {pretrained}) ===\n")


        ## 모델 선택
        # 모델 타입 설정(ResNet50 or ViT-S/16)
        train_dataset.set_model_type(model_type)
        test_dataset.set_model_type(model_type)

        if model_name == "ResNet50":
            # 모델 객체 생성
            model = ResNet50(num_classes = 10).to(device)
            # wehight 적용
            if pretrained:
                print("🔹 Applying Pre-trained Weights...")
                pretrained_weights = torch.hub.load_state_dict_from_url(
                    "https://download.pytorch.org/models/resnet50-19c8e357.pth", 
                    map_location = device
                )
                model.load_state_dict(pretrained_weights, strict = False)
                # FC Layer 수정 (1000 → 10 classes for CIFAR-10)
                model.fc = nn.Linear(in_features = 2048, out_features = 10).to(device)

        elif model_name == "ViT-S/16":
            # 모델 객체 생성
            model = timm.create_model("vit_small_patch16_224", pretrained = pretrained, num_classes = 10).to(device)
            # wehight 적용
            if pretrained:
                print("🔹 Applying Pre-trained Weights...")
                # FC Layer 수정 (1000 → 10 classes for CIFAR-10)
                model.head = nn.Linear(in_features = model.head.in_features, out_features = 10).to(device)


        ## 손실 함수, 옵티마이저, 스케쥴러 초기화
        criterion = loss_function()
        optimizer = optimizer_class(model.parameters(), lr = initial_lr)
        scheduler = get_scheduler(optimizer, epochs, warmup_epochs)


        ## 학습 및 평가 루프
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            f.write(f"Epoch {epoch + 1}/{epochs}\n")

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
                "Top-5 Error": test_metrics["top_k_error"] * 100,
                "Precision": test_metrics["precision"],
                "Recall": test_metrics["recall"],
                "F1 Score": test_metrics["f1"],
            }

            log_msg = "  " + ", ".join([f"{key}: {value:.4f}" for key, value in eval_results.items()])
            print(log_msg)
            f.write(log_msg + "\n")

            # Learning Rate Scheduler 업데이트
            scheduler.step()

        ## 모델 저장
        save_filename = f"{model_name.lower()}_{'pretrained' if pretrained else 'scratch'}.pth"
        save_path = os.path.join(cfg.model.save_dir, save_filename)
        os.makedirs(cfg.model.save_dir, exist_ok=True)
        torch.save(model.state_dict(), save_path)

        log_msg = f"✅ 모델 저장 완료: {save_path}"
        print(log_msg)
        f.write(log_msg + "\n")