# import wandb  
from tqdm import tqdm
import os

import torch
import torch.nn
from torch.optim.lr_scheduler import LambdaLR

import omegaconf
import argparse

from transformers import AutoModel
from transformers import set_seed
from accelerate import Accelerator  # Accelerate 추가

# model과 data에서 정의된 custom class 및 function을 import합니다.
"""
여기서 import 하시면 됩니다. 
"""
from utils import load_config, get_model_name, set_logger, wandb_logger
from data import get_dataloader
from model import BERTForClassification, ModernBERTForClassification

# 재현성을 위한 시드 고정
set_seed(42)



# Scheduler 정의 (constant learning rate)
def get_scheduler(optimizer, scheduler_type="constant"):
    return LambdaLR(optimizer, lr_lambda = lambda epoch: 1.0)


# 평가 metric
def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)


def train_iter(model, inputs, optimizer, accelerator, accum_step):
    inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}

    # Mixed Precision 지원
    with accelerator.autocast():
        logits, loss = model(**inputs)

    # Gradient Accumulation
    loss = loss / accum_step
    accelerator.backward(loss)

    # Gradient Accumulation Step 단위로 Optimizer 업데이트
    if accelerator.sync_gradients:
        optimizer.step()
        optimizer.zero_grad()

    # Accuracy 계산
    accuracy = calculate_accuracy(logits, inputs['label'])
    # wandb.log({'train_loss': loss.item(), 'train_accuracy': accuracy})  # 필요하면 활성화
    return loss.item(), accuracy


def valid_iter(model, inputs, accelerator):
    inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}
    logits, loss = model(**inputs)

    accuracy = calculate_accuracy(logits, inputs['label'])
    # wandb.log({'val_loss': loss.item(), 'val_accuracy': accuracy})  # 필요하면 활성화
    return loss.item(), accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Run the classification model.")
    parser.add_argument('--model_name', type=str, choices=["bert", "modernbert"], default="bert",
                        help="Choose the model: 'bert' (BERT-base) or 'modernbert' (ModernBERT-base)")
    parser.add_argument("--accum_step", type=int, default=1, help="Gradient accumulation steps")
    return parser.parse_args()


def main(configs: omegaconf.DictConfig):
    # accelerator 설정
    accelerator = Accelerator(gradient_accumulation_steps=configs.train.accum_step)
    
    # device 변수 추가 (에러 방지)
    device = accelerator.device
    print(f"Using device: {device}")

    # 로깅 설정
    logger = set_logger(configs)
    # wandb_logger(configs)

    # Load model
    model_name = get_model_name(configs.model.model_name)
    configs.model.model_name = model_name
    model_cls = ModernBERTForClassification if "modernbert" in model_name.lower() else BERTForClassification
    model = model_cls(configs)

    # Load data
    train_loader = get_dataloader(configs, 'train')
    valid_loader = get_dataloader(configs, 'valid')
    test_loader = get_dataloader(configs, 'test')

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.train.learning_rate)
    scheduler = get_scheduler(optimizer, configs.train.scheduler)
    
    # Hugging Face `accelerate` 준비 (모델, 옵티마이저, 데이터 로더 감싸기)
    model, optimizer, train_loader, valid_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, test_loader
    )

    # Train & Validation Loop
    best_valid_accuracy = 0.0
    checkpoint_dir = configs.checkpoint.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok = True)
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{'modernbert' if 'ModernBERT' in model_name else 'bert'}.safetensors")

    for epoch in range(configs.train.epochs):
        # Train
        model.train()
        total_train_loss, total_train_accuracy = 0, 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{configs.train.epochs}", ncols=100)):
            train_loss, train_accuracy = train_iter(model, batch, optimizer, accelerator, configs.train.accum_step)
            total_train_loss += train_loss
            total_train_accuracy += train_accuracy
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_accuracy = total_train_accuracy / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{configs.train.epochs} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")

        # Validation
        model.eval()
        total_valid_loss, total_valid_accuracy = 0, 0
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Validating Epoch {epoch+1}", ncols=100):
                valid_loss, valid_accuracy = valid_iter(model, batch, accelerator)
                total_valid_loss += valid_loss
                total_valid_accuracy += valid_accuracy

        avg_val_loss = total_valid_loss / len(valid_loader)
        avg_val_accuracy = total_valid_accuracy / len(valid_loader)
        logger.info(f"Epoch {epoch+1}/{configs.train.epochs} - Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}")

        # 성능 개선 시 checkpoint 갱신
        if avg_val_accuracy > best_valid_accuracy:
            best_valid_accuracy = avg_val_accuracy
            torch.save(model.state_dict(), checkpoint_path)  # safetensors 저장
            logger.info(f"Model updated: Saved at {checkpoint_path}")

        # 스케쥴러 업데이트
        scheduler.step()

    # Final test evaluation
    state_dict = torch.load(checkpoint_path, map_location=accelerator.device)
    model.load_state_dict(state_dict)  # 모델에 로드
    
    model.to(accelerator.device)
    model.eval()

    total_test_loss, total_test_accuracy = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", ncols=100):
            test_loss, test_accuracy = valid_iter(model, batch, accelerator)
            total_test_loss += test_loss
            total_test_accuracy += test_accuracy
    
    avg_test_loss = total_test_loss / len(test_loader)
    avg_test_accuracy = total_test_accuracy / len(test_loader)
    logger.info(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")

if __name__ == "__main__":
    args = parse_args()
    configs = load_config()
    configs.model.model_name = get_model_name(args.model_name) if args.model_name.lower() not in map(str.lower, ["bert-base-uncased", "answerdotai/modernbert-base"]) else args.model_name
    configs.train.accum_step = args.accum_step
    print(f"model: {configs.model.model_name}, accum_step: {configs.train.accum_step}")
    
    main(configs)
