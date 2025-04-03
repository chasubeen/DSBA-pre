import numpy as np
import os
import wandb
import json
import logging
import torch

from omegaconf import OmegaConf
from glob import glob

from exp_builder import training_dl, test_dl
from data_provider.factory import create_dataloader
from losses import create_criterion
from optimizers import create_optimizer
from models import create_model
from utils.log import setup_default_logging
from utils.utils import make_save, version_build, load_resume_model, Float32Encoder
from arguments import parser
from utils.tools import update_information
from utils.metrics import log_metrics

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

_logger = get_logger('train')


def main(cfg):
    ## [1] 기본 설정 초기화: 로깅, seed, accelerator
    # set logger setting
    setup_default_logging()
    # set seed
    set_seed(cfg.DEFAULT.seed)
    
    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
        mixed_precision             = cfg.TRAIN.mixed_precision
    )

    # 로깅 포맷 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    

    ## [2] 결과 저장 디렉토리 생성 및 버전 관리
    # make save directory
    savedir = os.path.join(cfg.RESULT.savedir, cfg.MODEL.modelname, 
                           cfg.DEFAULT.exp_name, cfg.DATASET.dataname)
    savedir = make_save(accelerator = accelerator, savedir=savedir, 
                        resume=cfg.TRAIN.resume)

    loaddir, savedir = version_build(accelerator = accelerator,
                                    logdir = savedir, 
                                    resume = cfg.TRAIN.resume)
    
    # set device
    if accelerator.is_local_main_process:
        _logger.info('Device: {}'.format(accelerator.device))


    ## [3] 데이터셋 로드 및 dataloader 생성
    # load and define dataloader
    trn_dataloader, valid_dataloader, test_dataloader, var = create_dataloader(
                datadir           = cfg.DATASET.datadir,
                dataname          = cfg.DATASET.dataname,
                modelname         = cfg.MODEL.modelname,
                scaler            = cfg.DATASET.scaler,
                batch_size        = cfg.DATASET.batch_size,
                shuffle           = cfg.DATASET.shuffle,
                num_workers       = cfg.DATASET.num_workers,
                pin_memory        = cfg.DATASET.pin_memory,
                drop_last         = cfg.DATASET.drop_last,
                seq_len           = cfg.DATASET.seq_len,
                label_len         = cfg.DATASET.label_len,
                pred_len          = cfg.DATASET.pred_len,
                split_rate        = cfg.DATASET.split_rate,
                time_embedding    = cfg.DATASET.time_embedding,
                del_feature       = cfg.DATASET.del_feature
                )
    

    ## [4] 모델 설정값(cfg.MODELSETTING) 업데이트 (입력 feature 수 등)
    # update cfg
    update_information(cfg        =  cfg, 
                       section    =  'MODELSETTING',
                       dim_in     =  var,
                       batch_size =  cfg.DATASET.batch_size,
                       seq_len    =  cfg.DATASET.seq_len,
                       pred_len   =  cfg.DATASET.pred_len,
                       label_len  =  cfg.DATASET.label_len
                       )
    

    ## [5] config 저장(yaml 형태로)
    # save configs
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
        print(OmegaConf.to_yaml(cfg))


    ## [6] wandb 설정 및 초기화
    # wandb
    if cfg.TRAIN.wandb.use:
        # initialize wandb
        wandb.init(name    = cfg.DEFAULT.exp_name, 
                   group   = cfg.TRAIN.wandb.exp_name,
                   project = cfg.TRAIN.wandb.project_name, 
                   entity  = cfg.TRAIN.wandb.entity, 
                   config  = OmegaConf.to_container(cfg)
                   )
    

    ## [7] 모델 생성
    # build Model
    model = create_model(
        modelname    = cfg.MODEL.modelname,
        params       = cfg.MODELSETTING
    )
    

    ## [8] 학습 재시작할 경우, 저장된 모델 로드
    # load weights
    if cfg.TRAIN.resume:
        load_resume_model(model=model, savedir=savedir, 
                          resume_num=cfg.TRAIN.resume_number)

    _logger.info('# of learnable params: {}'.format(
        np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))
    

    ## [9] 손실함수 및 옵티마이저 구성
    # set training
    criterion = create_criterion(loss_name=cfg.LOSS.loss_name)
    optimizer = create_optimizer(model=model, opt_name=cfg.OPTIMIZER.opt_name, lr=cfg.OPTIMIZER.lr, params=cfg.OPTIMIZER.params)
    

    ## [10] accelerator를 이용해 학습 객체 준비 (멀티 디바이스 대응)
    model, optimizer, trn_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, trn_dataloader, valid_dataloader, test_dataloader
    )
    

    ## [11] 학습 시작
    # fitting model
    training_dl(
        model                 = model, 
        trainloader           = trn_dataloader, 
        validloader           = valid_dataloader, 
        criterion             = criterion, 
        optimizer             = optimizer,
        accelerator           = accelerator, 
        epochs                = cfg.TRAIN.epoch,
        eval_epochs           = cfg.TRAIN.eval_epochs, 
        log_epochs            = cfg.TRAIN.log_epochs,
        log_eval_iter         = cfg.TRAIN.log_eval_iter,
        use_wandb             = cfg.TRAIN.wandb.use, 
        wandb_iter            = cfg.TRAIN.wandb.iter,
        ckp_metric            = cfg.TRAIN.ckp_metric, 
        label_len             = cfg.DATASET.label_len,
        pred_len              = cfg.DATASET.pred_len,
        savedir               = savedir,
        model_name            = cfg.MODEL.modelname,
        early_stopping_metric = cfg.TRAIN.early_stopping_metric,
        early_stopping_count  = cfg.TRAIN.early_stopping_count,
        lradj                 = cfg.TRAIN.lradj,
        learning_rate         = cfg.OPTIMIZER.lr,
        model_config          = cfg.MODELSETTING
    )


    ## [12] best model 로드
    # load best checkpoint weights
    model.load_state_dict(torch.load(os.path.join(savedir, 
                                                  'best_model.pt')))
    

    ## [13] 테스트 진행
    # test results
    fine_tuning_test_metrics = test_dl(
        accelerator   = accelerator,
        model         = model, 
        dataloader    = test_dataloader, 
        criterion     = criterion, 
        log_interval  = cfg.TRAIN.log_eval_iter, 
        label_len     = cfg.DATASET.label_len,
        pred_len      = cfg.DATASET.pred_len,
        name          = 'TEST',
        savedir       = savedir,
        model_name    = cfg.MODEL.modelname,
        model_config  = cfg.MODELSETTING,
        return_output = cfg.TRAIN.return_output
    )


    ## [14] 테스트 결과 저장 및 출력
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        log_metrics(cfg.MODEL.modelname, cfg.DATASET.dataname, 
                    fine_tuning_test_metrics, _logger)
        json.dump(fine_tuning_test_metrics, open(os.path.join(savedir, 
                                                              f'{cfg.MODEL.modelname}_{cfg.DATASET.dataname}_test_results.json'),'w'), 
                                                              indent='\t', cls=Float32Encoder)
        if cfg.TRAIN.del_pt_file:
            pt_files = glob(os.path.join(savedir, "*.pt"))
            for pt_file in pt_files:
                try:
                    os.remove(pt_file)
                except Exception as e:
                    _logger.warning(f"Failed to delete {pt_file}: {e}")

    print('\n🎉 Done 🎉 \n')

if __name__=='__main__':
    cfg = parser()
    main(cfg)
