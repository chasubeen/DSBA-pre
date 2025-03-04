import wandb
from tqdm import tqdm
import os
import time

import torch
import torch.nn
from torch.optim.lr_scheduler import LambdaLR

import omegaconf
from omegaconf import OmegaConf
import argparse

from transformers import set_seed

# model과 data에서 정의된 custom class 및 function을 import합니다.
"""
여기서 import 하시면 됩니다. 
"""
from utils import load_config, get_model_name, set_logger, wandb_logger
from data import get_dataloader
from model import BERTForClassification, ModernBERTForClassification

# 재현성을 위한 시드 고정
set_seed(42)

# Scheduler 정의(constant learning rate)
def get_scheduler(optimizer, scheduler_type = "constant"):
    return LambdaLR(optimizer, lr_lambda = lambda epoch: 1.0)

# 평가 metric
def calculate_accuracy(logits, label):
    preds = logits.argmax(dim=-1)
    correct = (preds == label).sum().item()
    total = label.size(0)
    return correct, total

def train_iter(model, inputs, optimizer, device, accum_step, step):
    inputs = {key: value.to(device) for key, value in inputs.items()}
    logits, loss = model(**inputs)
    
    loss = loss / accum_step
    loss.backward()
    
    if (step + 1) % accum_step == 0 or (step + 1 == len(inputs)):
        optimizer.step()
        optimizer.zero_grad()
    
    correct, num_samples = calculate_accuracy(logits, inputs['label'])
    
    return loss.item() * num_samples, correct, num_samples

def valid_iter(model, inputs, device):
    inputs = {key: value.to(device) for key, value in inputs.items()}
    logits, loss = model(**inputs)
    
    correct, num_samples = calculate_accuracy(logits, inputs['label'])
    
    return loss.item() * num_samples, correct, num_samples

# command line에서 모델 이름 받기
def parse_args():
    parser = argparse.ArgumentParser(description="Run the classification model.")
    parser.add_argument('--model_name', type=str, choices=["bert", "modernbert"], default="bert",
                        help="Choose the model: 'bert' (BERT-base) or 'modernbert' (ModernBERT-base)")
    parser.add_argument("--accum_step", type=int, default=1, help="Gradient accumulation steps")
    return parser.parse_args()

def main(configs: omegaconf.DictConfig):

    ## Set device
    torch.manual_seed(configs.train.seed) # 실험 재현성을 위한 시드 고정
    torch.cuda.manual_seed(configs.train.seed)
    torch.cuda.manual_seed_all(configs.train.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 디바이스 설정
    device = torch.device(configs.torch.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 로깅 설정
    logger = set_logger(configs)
    wandb_logger(configs)

    ## Load model
    model_name = get_model_name(configs.model.model_name)  # 모델 키 변환 적용
    configs.model.model_name = model_name  # config 업데이트
    model_cls = ModernBERTForClassification if "modernbert" in model_name.lower() else BERTForClassification
    model = model_cls(configs).to(device)

    ## Load data
    train_loader = get_dataloader(configs, 'train')
    valid_loader = get_dataloader(configs, 'valid')
    test_loader = get_dataloader(configs, 'test')

    ## Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.train.learning_rate)
    scheduler = get_scheduler(optimizer, configs.train.scheduler)
    
    ## Train & Validation Loop
    best_valid_accuracy = 0.0
    checkpoint_dir = configs.checkpoint.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok = True)
    checkpoint_path = os.path.join(checkpoint_dir, 
                                   f"best_model_{'modernbert' if 'ModernBERT' in model_name else 'bert'}.pt")

    for epoch in range(configs.train.epochs):
        # Train
        model.train()
        total_train_loss, total_correct, total_samples = 0, 0, 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(train_loader, 
                                     desc = f"Epoch {epoch+1}/{configs.train.epochs}", 
                                     ncols = 100)):
            train_loss, correct, num_samples = train_iter(model, batch, optimizer, device, configs.train.accum_step, step)
            total_train_loss += train_loss
            total_correct += correct
            total_samples += num_samples
            
            if step % 100 == 0:
                wandb.log({'train_loss': total_train_loss / total_samples, 'train_accuracy': total_correct / total_samples})
        
        avg_train_loss = total_train_loss / total_samples  
        avg_train_accuracy = total_correct / total_samples
        
        logger.info(f"Epoch {epoch+1}/{configs.train.epochs} - Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Epoch {epoch+1}/{configs.train.epochs} - Train Accuracy: {avg_train_accuracy:.4f}")

        # Validation
        model.eval()
        total_valid_loss, total_correct, total_samples = 0, 0, 0
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, 
                              desc = f"Validating Epoch {epoch+1}", 
                              ncols = 100):
                valid_loss, correct, num_samples = valid_iter(model, batch, device)
                total_valid_loss += valid_loss
                total_correct += correct
                total_samples += num_samples
        
        avg_val_loss = total_valid_loss / total_samples  
        avg_val_accuracy = total_correct / total_samples
        
        logger.info(f"Epoch {epoch+1}/{configs.train.epochs} - Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"Epoch {epoch+1}/{configs.train.epochs} - Validation Accuracy: {avg_val_accuracy:.4f}")
        
        wandb.log({'val_loss': avg_val_loss, 'val_accuracy': avg_val_accuracy})

        # 성능 개선 시 checkpoint 갱신
        if avg_val_accuracy > best_valid_accuracy:
            best_valid_accuracy = avg_val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Model updated: Saved at {checkpoint_path}")
        
        # 스케쥴러 업데이트
        scheduler.step()

    ## Final test evaluation
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    total_test_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc = "Testing", ncols = 100):
            test_loss, correct, num_samples = valid_iter(model, batch, device)
            total_test_loss += test_loss
            total_correct += correct
            total_samples += num_samples
    
    avg_test_loss = total_test_loss / total_samples  
    avg_test_accuracy = total_correct / total_samples
    
    logger.info(f"Test Loss: {avg_test_loss:.4f}")
    logger.info(f"Test Accuracy: {avg_test_accuracy:.4f}")


if __name__ == "__main__":
    args = parse_args()
    configs = load_config()
    
    # model 갱신
    configs.model.model_name = get_model_name(args.model_name) if args.model_name.lower() not in map(str.lower, ["bert-base-uncased", "answerdotai/modernbert-base"]) else args.model_name
    # accum_step 갱신
    configs.train.accum_step = args.accum_step
    # effective batch size 계산
    configs.train.effective_batch_size = configs.data.batch_size.train * configs.train.accum_step
    
    print(f"model: {configs.model.model_name}, accum_step: {configs.train.accum_step}, effective batch size: {configs.train.effective_batch_size}")
    
    main(configs)