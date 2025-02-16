import wandb
from tqdm import tqdm
import os

import torch
import torch.nn
from torch.optim.lr_scheduler import LambdaLR
import omegaconf
from omegaconf import OmegaConf

import argparse

from transformers import set_seed
set_seed(42)

# model과 data에서 정의된 custom class 및 function을 import합니다.
"""
여기서 import 하시면 됩니다. 
"""
from utils import load_config, get_model_name, set_logger, wandb_logger
from data import get_dataloader
from model import BERTForClassification, ModernBERTForClassification


# Scheduler 정의 (constant learning rate)
def get_scheduler(optimizer, scheduler_type = "constant"):
    return LambdaLR(optimizer, lr_lambda = lambda epoch: 1.0)


def train_iter(model, inputs, optimizer, device, epoch):
    inputs = {key: value.to(device) for key, value in inputs.items()}
    logits, loss = model(**inputs)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    accuracy = calculate_accuracy(logits, inputs['label'])
    wandb.log({'train_loss': loss.item(), 'train_accuracy': accuracy})
    return loss, accuracy


def valid_iter(model, inputs, device, log_samples=False):
    inputs = {key: value.to(device) for key, value in inputs.items()}
    logits, loss = model(**inputs)
    
    accuracy = calculate_accuracy(logits, inputs['label'])   
    wandb.log({'val_loss': loss.item(), 'val_accuracy': accuracy})
    
    # 오분류된 sample에 대한 logging 추가
    if log_samples:
        incorrect_preds = (logits.argmax(dim=-1) != inputs["label"])  
        incorrect_samples = [inputs["input_ids"][i].tolist() for i in range(len(incorrect_preds)) if incorrect_preds[i]]  
        wandb.log({'incorrect_samples': incorrect_samples})
    return loss, accuracy

def calculate_accuracy(logits, label):
    preds = logits.argmax(dim = -1)
    correct = (preds == label).sum().item()
    return correct / label.size(0)

# command line에서 모델 이름 받기
def parse_args():
    parser = argparse.ArgumentParser(description = "Run the classification model.")
    parser.add_argument('--model_name', type=str, choices=["bert", "modernbert"], default="bert",
                        help="Choose the model: 'bert' (BERT-base) or 'modernbert' (ModernBERT-base)")
    return parser.parse_args()

def main(configs: omegaconf.DictConfig):
    ## Set device
    # 실험 재현성을 위한 시드 고정
    torch.manual_seed(configs.train.seed)
    torch.cuda.manual_seed(configs.train.seed)
    torch.cuda.manual_seed_all(configs.train.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(11/24) 

    # 디바이스 설정
    device = torch.device(configs.torch.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 로깅 설정
    logger = set_logger(configs)
    wandb_logger(configs)

    ## Load model
    model_name = get_model_name(configs.model.model_name)  # 모델 키 변환 적용
    configs.model.model_name = model_name  # config 업데이트
    # 모델 선택
    if "modernbert" in configs.model.model_name.lower():
        model = ModernBERTForClassification(configs)
    else:
        model = BERTForClassification(configs)
    model.to(device)

    ## Load data
    train_loader = get_dataloader(configs, 'train')
    valid_loader = get_dataloader(configs, 'valid')
    test_loader = get_dataloader(configs, 'test')

    ## Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.train.learning_rate)
    
    # Get scheduler from config.yaml
    scheduler = get_scheduler(optimizer, configs.train.scheduler)

    best_valid_accuracy = 0.0
    checkpoint_dir = configs.checkpoint.checkpoint_dir  
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_name = f"best_model_{'modernbert' if 'modernbert-base' in model_name else 'bert'}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    ## Train & validation for each epoch
    for epoch in range(configs.train.epochs):
        # Train
        model.train()
        total_train_loss, total_train_accuracy = 0, 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{configs.train.epochs}", ncols=100):
            train_loss, train_accuracy = train_iter(model, batch, optimizer, device, epoch)  
            total_train_loss += train_loss.item()
            total_train_accuracy += train_accuracy
        
        logger.info(f"Epoch {epoch+1}/{configs.train.epochs} - Train Loss: {total_train_loss / len(train_loader)}")
        logger.info(f"Epoch {epoch+1}/{configs.train.epochs} - Train Accuracy: {total_train_accuracy / len(train_loader)}")

        # Validation
        model.eval()
        total_valid_loss, total_valid_accuracy = 0, 0
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Validating Epoch {epoch+1}", ncols=100):
                valid_loss, valid_accuracy = valid_iter(model, batch, device)
                total_valid_loss += valid_loss.item()
                total_valid_accuracy += valid_accuracy
        logger.info(f"Epoch {epoch+1}/{configs.train.epochs} - Validation Loss: {total_valid_loss / len(valid_loader)}")
        logger.info(f"Epoch {epoch+1}/{configs.train.epochs} - Validation Accuracy: {total_valid_accuracy / len(valid_loader)}")
        
        # 성능 개선 시 checkpoint 갱신
        if total_valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = total_valid_accuracy            
            torch.save(model.state_dict(), checkpoint_path)  
            logger.info(f"Model updated: Saved at {checkpoint_path}")
        
        # 스케쥴러 업데이트
        scheduler.step()

    ## Final test evaluation
    model.eval()
    model.load_state_dict(torch.load(checkpoint_path)) 
    
    total_test_loss, total_test_accuracy = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", ncols=100):
            test_loss, test_accuracy = valid_iter(model, batch, device)
            total_test_loss += test_loss.item()
            total_test_accuracy += test_accuracy
    logger.info(f"Test Loss: {total_test_loss / len(test_loader)}")
    logger.info(f"Test Accuracy: {total_test_accuracy / len(test_loader)}")

if __name__ == "__main__":
    args = parse_args()
    configs = load_config()

    if args.model_name.lower() not in map(str.lower, ["bert-base-uncased", "answerdotai/modernbert-base"]):
        configs.model.model_name = get_model_name(args.model_name)
    else:
        configs.model.model_name = args.model_name  

    print(f"model: {configs.model.model_name}")  # 디버깅 출력
    main(configs)