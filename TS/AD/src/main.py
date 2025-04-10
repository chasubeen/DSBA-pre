### ===== 라이브러리 import =====
# 필수 라이브러리
import numpy as np
import os
import wandb
import logging
import torch

# 설정 파일 관리
from omegaconf import OmegaConf
from glob import glob

# 사용자 정의 모듈(파이프라인 구성요소)
from exp_builder_dl import training_dl, test_dl
from data_provider import create_dataloader
from losses import create_criterion
from optimizers import create_optimizer
from models import create_model

# logging 관리
from utils.log import setup_default_logging
from utils.utils import version_build, save_directory, load_resume_model
from utils.tools import update_information
from arguments import parser

# 병렬 GPU 학습
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# set logger
_logger = get_logger('train')


### ===== 실행 =====
def main(cfg):
    # set logger setting
    setup_default_logging()
    # set seed
    set_seed(cfg.DEFAULT.seed)
    
    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = cfg.TRAIN.grad_accum_steps,
        mixed_precision             = cfg.TRAIN.mixed_precision
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Save directory 설정(version 경로 정리)
    savedir = save_directory(
                        savedir   = cfg.RESULT.savedir,
                        dataname  = cfg.DATASET.dataname,
                        modelname = cfg.MODEL.modelname,
                        bank_name = cfg.DATASET.bank_name,
                        target    = cfg.DATASET.target,
                        exp_name  = cfg.DEFAULT.exp_name
                    )
    loaddir, savedir = version_build(accelerator = accelerator,
                                    logdir = savedir, 
                                    resume = cfg.TRAIN.resume)
    
    # set device(multi GPU 환경)
    if accelerator.is_local_main_process:
        _logger.info('Device: {}'.format(accelerator.device))

    # load and define dataloader
    # 슬라이딩 윈도우 기반
    trn_dataloader, val_dataloader, tst_dataloader, var = create_dataloader(
        datadir = cfg.DATASET.datadir,
        dataname = cfg.DATASET.dataname,
        modelname = cfg.MODEL.modelname,
        scaler = cfg.DATASET.scaler,
        batch_size = cfg.DATASET.batch_size,
        shuffle = cfg.DATASET.shuffle,
        num_workers = cfg.DATASET.num_workers,
        pin_memory = cfg.DATASET.pin_memory,
        drop_last = cfg.DATASET.drop_last,
        seq_len = cfg.DATASET.seq_len,
        stride_len = cfg.DATASET.stride_len,
        target = cfg.DATASET.target,
        val_split_rate = cfg.DATASET.val_split_rate,
        bank_name = cfg.DATASET.bank_name,
        merge_bank = cfg.DATASET.merge_bank,
        time_embedding = cfg.DATASET.time_embedding,
        del_feature = cfg.DATASET.del_feature
    )

    
    # update cfg
    update_information(cfg        =  cfg, 
                       section    =  'MODELSETTING',
                       dim_in     =  var,
                       batch_size =  cfg.DATASET.batch_size
                    )
    
    # save configs
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
        print(OmegaConf.to_yaml(cfg))

    # wandb
    if cfg.TRAIN.wandb.use:
        # initialize wandb
        wandb.init(name    = cfg.DEFAULT.exp_name, 
                   group   = cfg.TRAIN.wandb.exp_name,
                   project = cfg.TRAIN.wandb.project_name, 
                   entity  = cfg.TRAIN.wandb.entity, 
                   config  = OmegaConf.to_container(cfg))
        

    # build Model
    model = create_model(
        modelname    = cfg.MODEL.modelname,
        params       = cfg.MODELSETTING
        )
    
    # wandb
    if cfg.TRAIN.wandb.use:
        wandb.watch(model)

    # load weights
    if cfg.TRAIN.resume:
        load_resume_model(model=model, savedir=savedir, resume_num=cfg.TRAIN.resume_number)
    
    # set training
    _logger.info('# of learnable params: {}'.format(np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))
    criterion = create_criterion(loss_name=cfg.LOSS.loss_name)
    optimizer = create_optimizer(model=model, opt_name=cfg.OPTIMIZER.opt_name, lr=cfg.OPTIMIZER.lr, params=cfg.OPTIMIZER.params)
    
    model, optimizer, trn_dataloader, val_dataloader, tst_dataloader = accelerator.prepare(
        model, 
        optimizer, 
        trn_dataloader, val_dataloader, tst_dataloader
    )

    training_dl(
        model                 = model, 
        trn_dataloader        = trn_dataloader, 
        val_dataloader        = val_dataloader, 
        criterion             = criterion, 
        optimizer             = optimizer,
        accelerator           = accelerator, 
        savedir               = savedir,
        epochs                = cfg.TRAIN.epoch,
        eval_epochs           = cfg.TRAIN.eval_epochs, 
        log_epochs            = cfg.TRAIN.log_epochs,
        log_eval_iter         = cfg.TRAIN.log_eval_iter,
        use_wandb             = cfg.TRAIN.wandb.use, 
        wandb_iter            = cfg.TRAIN.wandb.iter,
        ckp_metric            = cfg.TRAIN.ckp_metric,
        model_name            = cfg.MODEL.modelname,
        early_stopping_metric = cfg.TRAIN.early_stopping_metric,
        early_stopping_count  = cfg.TRAIN.early_stopping_count,
        lradj                 = cfg.TRAIN.lradj,
        learning_rate         = cfg.OPTIMIZER.lr,
        model_config          = cfg.MODELSETTING
    )

    # load best checkpoint weights
    model.load_state_dict(torch.load(os.path.join(savedir, 'best_model.pt')))
    
    # test results
    test_metrics = test_dl(
        model         = model, 
        dataloader    = tst_dataloader,     
        criterion     = criterion, 
        accelerator   = accelerator,
        log_interval  = cfg.TRAIN.log_eval_iter, 
        savedir       = savedir,
        model_config  = cfg.MODELSETTING,
        model_name    = cfg.MODEL.modelname,
        name          = 'TEST',
        return_output = cfg.TRAIN.return_output,
        plot_result   = cfg.TRAIN.plot_result
    )

    # 결과 출력
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("\n📊 [Final Test Metrics]")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
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