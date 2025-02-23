## **🤗 NLP 사전학습(2)**

### **🎯 Overview**

1. **주제:**
    - 지난 주 작성한 코드에 gradient accumulation 기법을 적용한 실험
        - 우선, 기존 코드를 torch 기반으로 gradient accumulation을 구현
        - 이후 Huggingface Accelerate를 활용한 방식으로 실험 진행
    - **목표:** 64, 256, 1024의 배치 사이즈 중 최적의 배치 사이즈 선정
        
        > 실험 결과를 통해 gradient accumulation이 모델 학습에 미치는 영향 및 각 배치 사이즈의 성능 및 학습 효율성 평가
        > 
2. **학습 모델:**
    - BERT-base-uncased
    - ModernBERT-base
3. **실험 setting:**
    - Gradient Accumulation 적용:
        - Torch 기반(`main.py`)
        - Huggingface Accelerate 기반(`main_accelerate.py`)
    - **실험 배치 사이즈:** `64`, `256`, `1024`
    - **학습 Epoch:** `5`
4. **결과:**
    - 각 배치 사이즈 별로 Torch와 Accelerate 방식 비교
    - 학습 정확도, 검증 정확도 및 학습 소요 시간 등을 기록하여 최적의 배치 사이즈 선정

### **📂 파일 구조**
```bash
exp_2/
│── configs/                 
│   ├── config.yaml          # 실험 파라미터 및 gradient accumulation 설정 포함
│── src/                     
│   ├── utils.py             # 설정 로드 및 유틸 함수
│   ├── data.py              # Dataset 및 DataLoader 정의
│   ├── model.py             # BERT 기반 분류 모델
│   ├── main.py              # Torch 기반 gradient accumulation 학습 스크립트
│   ├── main_accelerate.py   # Huggingface Accelerate 기반 학습 스크립트
│── logs/                    # 학습 로그 및 wandb 기록 
│── checkpoints/             # 모델 체크포인트 저장 (용량 문제로 GitHub에 업로드하지 않음)
```

### **🛠️ Descriptions**

**📌 설정 및 유틸(**`configs/config.yaml`, `src/utils.py`)

- 실험 관련 주요 파라미터(배치 사이즈, accumulation steps, learning rate, epoch 등)를 설정 파일에 정의하여 관리
    - 재현성을 위한 랜덤 시드 고정 및 환경 변수 처리 수행

---

**📌 데이터 로딩 및 전처리(**`src/data.py`)

- 데이터셋을 로드하고 토큰화 후, train/validation/test로 분할하여 DataLoader를 구성
- Gradient Accumulation을 적용하기 위한 미니 배치 구성 포함
    - 우선적으로 `batch_size = 64` 지정
    - 이후 gradient accumulation을 통해 effective batch size가 결정됨

---

**📌 모델 정의(**`src/model.py`)

- 기존 사전학습 모델(BERT 기반 등)을 활용하여 fine-tuning을 위한 모델 구조를 정의

---

**📌 학습 및 검증**

- gradient accumulation 실험에 최적화된 모델 파라미터 업데이트 로직 포함
- Torch 기반(`src/train_torch.py`)
    - 기존 코드에 gradient accumulation 로직을 직접 구현하여 실험
    - 배치 사이즈(64, 256, 1024)에 따른 성능 및 학습 시간 비교
    - wandb logging을 통해 실험 모니터링 및 결과 기록
- Huggingface Accelerate 기반(`src/train_accelerate.py`)
    - Accelerate 라이브러리를 사용해 간편하게 gradient accumulation을 구현 가능
    - pytorch와 동일한 작업 수행
        - 코드 구동 여부를 확인하기 위해 BERT model, batch_size = 64에 대해서만 실험 진행

### **📊 실험 결과**

| Model | (effective)<br> batch size | Train Accuracy<br>(best) | Validation Accuracy<br>(best) | Test Accuracy<br>(best) | Training Durations |
| --- | --- | --- | --- | --- | --- |
| BERT | 64 | 0.9892 | **0.9009** | 0.9011 | 21m 14s |
| BERT | 256 | **0.9908** | 0.8999 | **0.9059** | 21m 01s |
| BERT | 1024 | 0.9743 | 0.9007 | 0.9001 | 21m 10s |
| ModernBERT | 64 | 0.9942 | **0.9185** | **0.9203** | 30m 55s |
| ModernBERT | 256 | **0.9948** | 0.9181 | 0.9114 | 30m 09s |
| ModernBERT | 1024 | 0.9931 | 0.9136 | 0.9193 | 29m 51s |

**[bert]**  
![image](https://github.com/user-attachments/assets/203ec55d-5446-498e-9dd0-c5192f5f1824)  

**[modernbert]**  
![image](https://github.com/user-attachments/assets/50444774-be06-4cdc-9c7c-7cab612c57ef)  


- 학습 안정성
    - BERT
        - 최적의 배치 사이즈: **256**
        - 파라미터 수가 상대적으로 적어, 큰 배치 사이즈에서도 사전 학습된 정보를 안정적으로 유지하며 업데이트가 이루어진다고 해석 가능
    - ModernBERT
        - 최적의 배치 사이즈: **64**
        - ModernBERT는 BERT에 비해 파라미터 수가 많아, 큰 배치 사이즈로 fine-tuning을 진행하면 한 번의 업데이트에서 기존 사전 학습 파라미터가 과도하게 변경될 가능성이 존재함  
            → 모델의 안정성을 저해할 수 있으므로, 보다 작은 배치 사이즈가 fine-tuning 시 효과적임을 시사
    
    ⇒ 적절한 effective batch size가 모델의 안정적인 업데이트와 일반화에 긍정적인 영향을 미친다.  
- 학습 속도  
    - 두 모델 모두 배치 사이즈에 따른 학습 시간 차이는 크지 않음
    - 그러나 ModernBERT의 경우 전체 학습 시간이 약 30분으로 BERT보다 소폭 긺  
        - 모델 복잡도 및 파라미터 수 차이와 연관될 수 있음

### 📝 **결과 분석 및 결론**
1. **Gradient Accumulation 효과:**
    - 작은 배치 사이즈에서도 충분한 gradient 업데이트를 통한 학습 안정성 확보
    - Torch와 Accelerate 방식 간의 성능 및 학습 시간 차이 분석
2. **최적의 Batch Size 선정:**
    - 64, 256, 1024 배치 사이즈 실험 결과를 바탕으로 최종 모델 성능과 학습 효율의 균형을 고려하여 선정
3. **실험 세팅 개선 사항:**
    - 동일한 학습 epoch과 learning rate를 모든 배치 사이즈에 대해 사용하면, 배치 사이즈에 따른 최적의 하이퍼파라미터 조정이 반영되지 않아 결과가 편향될 가능성이 있음
        - 예를 들어, 큰 배치 사이즈는 일반적으로 더 안정적인 gradient 추정을 가능하게 하기에 더 큰 learning rate를 사용해도 문제가 없거나 더 빠른 수렴을 기대할 수 있음
        - 반대로 작은 배치 사이즈는 노이즈가 많아 상대적으로 작은 learning rate가 필요할 수 있음
    - 또한, 배치 사이즈에 따라 1 epoch 내에서 실제 weight update 횟수가 달라지므로, 전체 학습 과정에서의 업데이트 횟수나 효과도 달라질 수 있음
    
    ⇒ 따라서 이번 실험 설정은 배치 사이즈 변화에 따른 추가 튜닝 없이 단순 비교를 위한 기준점으로는 유용하지만, 최적의 성능을 내기 위한 세부 조정 측면에서는 한계가 존재할 수 있음
    
    ⇒ 이를 개선하기 위해서는 각 배치 사이즈에 대해 적절한 learning rate 조정(ex. linear scaling rule 적용)이나, 학습 epoch을 재설정하는 등의 추가 실험이 필요할 수 있음
