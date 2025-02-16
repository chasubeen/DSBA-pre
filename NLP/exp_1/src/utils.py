import os
import sys
import torch
import logging
import omegaconf
from omegaconf import OmegaConf
import wandb

def load_config(config_path: str = "exp_1/configs/config.yaml") -> omegaconf.DictConfig:
    """
    config.yaml 파일을 불러와서 OmegaConf 객체로 반환하는 함수
    """
    config = OmegaConf.load(config_path)

    # 환경변수에서 W&B API Key 불러오기
    wandb_api_key = os.getenv(config.wandb.api_key_env, None)
    if wandb_api_key is None:
        raise ValueError("W&B API Key가 설정되지 않았습니다.")
    else:
        os.environ["WANDB_API_KEY"] = wandb_api_key  # W&B 실행을 위해 환경변수 설정
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

    # 지원되지 않는 경우 기본 모델 반환 (경고 출력)
    else:
        logging.warning(f"지원되지 않는 모델 키 '{model_key}'가 입력되었습니다. 기본 모델 'bert-base-uncased'로 설정됩니다.")
        return "bert-base-uncased"

def set_logger(configs):
    """
    로깅 설정 함수
    """
    log_file = configs.logging.log_file.format(model_name = configs.model.model_name)

    # 로그 디렉토리 생성
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok = True)

    # 로그 설정
    logging.basicConfig(
        filename = log_file,
        level = logging.INFO,
        format = "%(asctime)s:%(levelname)s:%(message)s"
    )

    # 콘솔 출력 추가
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
    if configs.logging.use_wandb:
        wandb.init(
            project = configs.wandb.project,
            name = configs.wandb.experiment_name.format(model_name=configs.model.model_name),
            config = OmegaConf.to_container(configs.wandb.config, resolve=True),
            dir=configs.wandb.dir
        )
