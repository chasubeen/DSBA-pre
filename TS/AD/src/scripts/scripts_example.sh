export CUDA_VISIBLE_DEVICES=0

TARGETS=("case1" "case2" "case3")
BANKS=("JFG" "IUR" "MIQ")
MODELS=("LSTM_AE")

for TARGET in "${TARGETS[@]}"; do
    for BANK in "${BANKS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            echo "▶️ Running MODEL=$MODEL | TARGET=$TARGET | BANK=$BANK"

            python main.py \
                --model_name "$MODEL" \
                --default_cfg ./configs/default_setting.yaml \
                --model_cfg ./configs/model_setting.yaml \
                --opts "DATASET.target=['$TARGET']" "DATASET.bank_name=['$BANK']"
        done
    done
done