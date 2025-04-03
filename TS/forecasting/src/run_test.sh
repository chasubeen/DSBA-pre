# model_name=DLinear

# python ./src/main.py \
#     --model_name $model_name \
#     --default_cfg ./src/configs/default_setting.yaml \
#     --model_cfg ./src/configs/model_setting.yaml \


model_name=DLinear_RevIN
for data_name in ETTh1 ETTh2 ETTm1 ETTm2; do
    python ./src/main.py \
        --model_name $model_name \
        --default_cfg ./src/configs/default_setting.yaml \
        --model_cfg ./src/configs/model_setting.yaml \
        DEFAULT.exp_name ${model_name}_${data_name} \
        TRAIN.wandb.exp_name $model_name \
        DATASET.dataname $data_name \
        DATASET.pred_len $pred_len
done