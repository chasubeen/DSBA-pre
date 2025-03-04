import os
import sys
import torch
import logging
import omegaconf
from omegaconf import OmegaConf
import wandb


def load_config(config_path: str = "exp_2/configs/config.yaml") -> omegaconf.DictConfig:
    """
    config.yaml 파일을 불러와서 OmegaConf 객체로 반환하는 함수
    """

    # config.yaml 로드
    config = OmegaConf.load(config_path)

    # 환경변수에서 W&B API Key 불러오기
    wandb_api_key = os.getenv("WANDB_API_KEY", None)
    if wandb_api_key is None:
        raise ValueError("W&B API Key가 설정되지 않았습니다.")
    

    # W&B API Key 환경변수 설정 (이제 config를 로드한 후 실행)
    os.environ["WANDB_API_KEY"] = wandb_api_key

    return config


def get_model_name(model_key: str) -> str:
    """
    모델 키에 따라 실제 모델 이름을 반환하는 함수
    """
    model_mapping = {
        "bert": "bert-base-uncased",
        "modernbert": "answerdotai/ModernBERT-base"
    }
    model_key_lower = model_key.lower()

    # 이미 full model name이면 그대로 반환
    if model_key_lower in model_mapping:
        return model_mapping[model_key_lower]

    # 모델 키가 등록된 경우 변환 후 반환
    elif model_key_lower in map(str.lower, model_mapping.values()):
        return model_key

    # 지원되지 않는 경우 기본 모델 반환(경고 출력)
    else:
        logging.warning(f"지원되지 않는 모델 키 '{model_key}'가 입력되었습니다. 기본 모델 'bert-base-uncased'로 설정됩니다.")
        return "bert-base-uncased"


def set_logger(configs):
    """
    로깅 설정 함수 (effective_batch_size 적용)
    """
    log_file = configs.logging.log_file.format(
        model_name = configs.model.model_name,
        batch_size_train = configs.data.batch_size.train,  # 직접 전달
        accum_step = configs.train.accum_step  # 직접 전달
    )

    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    return logging.getLogger()


def wandb_logger(configs):
    """
    W&B 초기화
    """
    experiment_name = configs.wandb.experiment_name.format(
        model_name = configs.model.model_name,
        batch_size_train = configs.data.batch_size.train,  
        accum_step = configs.train.accum_step  
    )

    wandb.init(
        project = configs.wandb.project,
        name = experiment_name,
        config = {
            "model": configs.model.model_name,
            "optimizer": configs.wandb.config.optimizer,
            "lr": configs.wandb.config.lr,
            "batch_size": configs.train.effective_batch_size,  
            "epoch": configs.wandb.config.epoch,
            "max_seq_length": configs.wandb.config.max_seq_length
        },
        dir = configs.wandb.dir
    )