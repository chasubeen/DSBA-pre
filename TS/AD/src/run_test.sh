export CUDA_VISIBLE_DEVICES=0
python /pre-ts/Time-series-AD/src/main.py \
    --model_name LSTM_AE \
    --default_cfg /pre-ts/Time-series-AD/src/configs/default_setting.yaml \
    --model_cfg /pre-ts/Time-series-AD/src/configs/model_setting.yaml  \
