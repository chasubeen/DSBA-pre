## **🤗 NLP 사전학습(1)**

### **🎯 Overview**
1. **주제:**
- IMDB 감성 분류를 위한 `Encoder` 모델 비교 실험
    - [IMDB 감성 분석(긍정/부정 분류)](https://huggingface.co/datasets/stanfordnlp/imdb) 성능 비교
    - Transformer 기반 Encoder 모델의 문장 표현 방식이 성능에 미치는 영향 분석
    
    > 실험 결과를 바탕으로 각 모델의 특징 및 적용 가능성 평가
  
2. **학습 모델:** 
    - BERT-base-uncased
    - ModernBERT-base
3. **실험 setting:**
    - 데이터 분할: Train:Valid:Test = 8:1:1
    - 최대 학습 epoch: `5`
    - optimizer: `Adam`
    - lr: `5e-5`
    - max_len: `128`
    - scheduler: `constant`
    - 실험 재현을 위한 시드 고정
    
      ```python
      from transformers import set_seed
      set_seed(42)
      ```
    
4. **결과:**
    - 학습 중 매 epoch validation 진행, checkpoint 저장
    - 최적의 checkpoint에 대한 test 진행
    - wandb logging 필수

      
### **📂 파일 구조**
```
project/
│── configs/                 # 설정 관련 폴더
│   ├── config.yaml          # 설정 파일
│── src/                     # 핵심 코드 모듈
│   ├── utils.py             # 설정 로드 & 유틸 함수
│   ├── data.py              # Dataset & DataLoader 정의
│   ├── model.py             # BERT 기반 분류 모델
│   ├── main.py              # 학습 및 검증 실행
│── checkpoints/             # 저장된 모델 체크포인트
```

### **🛠️ Descriptions**

**📌 설정 및 유틸(**`configs/config.yaml`, `src/utils.py`**)**

- `OmegaConf`를 활용한 설정 로드 및 환경 변수 처리
    - 모델, 데이터, 학습 관련 공통 설정 저장
        - `data`, `train`, `torch`, `logging` 등의 공통 환경 설정 유지
- `get_model_name()`을 통해 BERT와 ModernBERT의 모델명 자동 매핑
    
    ```python
    model_mapping = {
            "bert": "bert-base-uncased",
            "modernbert": "answerdotai/ModernBERT-base"
        }
    ```
    
- 로깅 및 실험 기록: `set_logger()`, `wandb_logger()`

---

**📌 데이터 로딩 및 전처리(**`src/data.py`**)**

- IMDB 데이터 로드 및 병합 → 분리
    - Train:Valid:Test = 8:1:1
- `AutoTokenizer`를 활용한 토큰화
    - `padding='max_length', truncation=True, max_length=128`
- `PyTorch Dataset` 및 `DataLoader` 정의
    - BERT와 ModernBERT의 `token_type_ids` 차이 처리
        
        ```python
        # BERT 모델일 때만 token_type_ids 포함
        if self.data_config.model.model_name.lower() == "bert-base-uncased":
            input_data["token_type_ids"] = torch.tensor(self.data["token_type_ids"][idx], dtype=torch.long)
        ```
        
    - `collate_fn()` 활용하여 batch tensor 변환
        ```
        Testing: Data Loading ...
        Map: 100%|███████████████████████████████████████████████| 40000/40000 [00:07<00:00, 5407.17 examples/s]
        >> SPLIT: train | Total Data Length: 40000
        Map: 100%|█████████████████████████████████████████████████| 5000/5000 [00:00<00:00, 5513.82 examples/s]
        >> SPLIT: valid | Total Data Length: 5000
        Map: 100%|█████████████████████████████████████████████████| 5000/5000 [00:00<00:00, 5890.03 examples/s]
        >> SPLIT: test | Total Data Length: 5000
        ✅ Train DataLoader Size: 5000 # 40000 / 8
        ✅ Valid DataLoader Size: 625 # 5000 / 8
        ✅ Test DataLoader Size: 1250 # 5000 / 4
        
        🔹 First batch sample:
        🔸 Input IDs shape: torch.Size([8, 128])
        🔸 Attention Mask shape: torch.Size([8, 128])
        🔸 Labels shape: torch.Size([8])
        ```
---

**📌 모델 정의**(`src/model.py`)

- `BERT` 및 `ModernBERT`의 분류 모델 구현
    - `token_type_ids` 활용 유무 반영
- 문장 표현 방식 차이점:
    - initial setting: `[CLS]`를 대표 토큰으로 학습
        
        ```python
        pooled_output = outputs.last_hidden_state[:, 0, :]
        ```
        
        → `[CLS]`가 충분한 정보를 담지 못할 수도 있음  
        → 실제로 BERT 초기 학습이 매우 불안정하였음(정확도: 50% 정도)  
        
    - 2nd_ver: 문장 전체 평균 (`torch.mean()`) 사용
        
        ```python
        pooled_output = torch.mean(outputs.last_hidden_state, dim = 1)
        ```
        
        → 문장 전체의 정보를 평균적으로 반영하기에 [CLS] token 하나보다 더 안정적일 가능성이 있음  
        (실제 성능 개선: 30%정도)  
        → BERT와 ModernBERT 간 구조적 차이를 줄이고, 성능 차이 최소화
        

---

**📌 학습 및 검증(**`src/main.py`**)**

- `config.yaml`을 로드하여 설정 적용
- `scheduler` 정의 및 적용
- 모델 및 데이터 로드 후 학습 진행
    - `train_iter()` & `valid_iter()`
- Validation 진행 후 Best Checkpoint 저장
    
    ```python
    if total_valid_accuracy > best_valid_accuracy:
        best_valid_accuracy = total_valid_accuracy
        torch.save(model.state_dict(), checkpoint_path)
    
    ```

### **📊 실험 결과**

| 모델 | Train Accuracy | Validation Accuracy | Test Accuracy |
| --- | --- | --- | --- |
| **BERT-base-uncased** | 0.9713 | 0.8854 | 0.8804 |
| **ModernBERT-base** | 0.9865 | 0.9114 | 0.909 |

⇒ ModernBERT가 약간 더 높은 성능을 보였음  
⇒ 기존 `[CLS]` 방식에서는 BERT 성능이 낮았으나, `torch.mean()` 적용 후 ModernBERT와 유사해짐

![image](https://github.com/user-attachments/assets/fea241f0-094c-4c73-bb6c-74c32d7356a4)
![image](https://github.com/user-attachments/assets/f49977b2-1ca8-4892-ad83-f0ae533742a5)

- BERT(blue), ModernBERT(purple)

### **📝 결과 분석**

**Papers:**  
- BERT
  - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)  
- ModernBERT
  - [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663)
  - [Reference_blog](https://blog.sionic.ai/modernbert)

---

**1️⃣ BERT vs ModernBERT: 주요 차이점**

|  | **BERT-base-uncased** | **ModernBERT-base** |
| --- | --- | --- |
| **Pooling 방식** | `pooler_output` 사용 (`tanh(W[CLS])`) | Pooling Layer 없음 |
| **문장 표현 방식** | `[CLS]` 토큰을 대표값으로 사용 | `last_hidden_state`에서 직접 특성 추출 |
| **구조적 특징** | 표준 Transformer Encoder | BERT 구조를 유지하면서 Pooling Layer 제거 |
| **모델 경량화** | 표준 BERT보다 약간 무거움 | Pooling Layer 제거로 연산량 감소 |
| **Fine-tuning 적용** | 특정 태스크에 맞게 `pooler_output` 학습 가능 | Pooling Layer가 없으므로 직접 특성 추출 필요 |
- BERT는 Pooling Layer를 통해 `[CLS]` 토큰을 추가적으로 가공하여 문장 표현을 생성함
- ModernBERT는 `pooler_output` 없이도 충분히 높은 성능을 유지하도록 설계됨

---

**2️⃣ 성능 차이가 발생한 원인**  
- BERT의 `[CLS]` 토큰
    - BERT는 `[CLS]` 토큰이 문장 전체의 의미를 담도록 학습되지만, 초기 학습 단계에서 충분한 의미를 가지지 못할 가능성이 있음
    - 일부 논문에서는 “[CLS] 토큰이 항상 최적의 문장 표현이 아닐 수 있다"는 점을 지적
- ModernBERT의 개선점:
    - ModernBERT에서는 Pooling Layer 없이도 충분한 표현력을 가질 수 있도록 설계되었다고 주장
        - 즉, ModernBERT는 BERT의 `[CLS]` 토큰을 따로 가공하지 않아도 성능을 유지할 수 있도록 설계됨
    - 실험에서도 BERT에서 `torch.mean()`을 적용하면 ModernBERT와 거의 동일한 성능을 보였음  
        → 이는 `[CLS]` 토큰이 문장 의미를 온전히 반영하지 않는다는 가설을 뒷받침함
        - `[CLS]`만 사용하면 특정 패턴(주로 첫 번째 단어)만 반영될 가능성이 높음
        - 결과적으로 BERT에서도 `pooler_output` 없이도 충분한 성능을 낼 수 있음을 입증

**3️⃣ 어떤 모델이 더 좋은가?**

- `ModernBERT`가 더 나은 경우
    - 연산 속도를 줄이면서도 BERT와 유사한 성능을 원할 때
    - pooling layer 없이도 높은 성능을 유지하고자 할 때
    - 학습 및 fine-tuning 과정에서 추가적인 pooling layer 없이 직접 특성을 추출하고 싶을 때
    - 배포 환경에서 효율적인 모델을 원할 때
        - 경량화된 모델 → 실시간 서비스에 유리
- 기존 `BERT`가 더 유리한 경우
    - 기존 파이프라인을 유지해야 할 때
    - 특정 태스크에서 `[CLS]` 토큰을 활용하는 것이 유리할 때
    - fine-tuning 시 추가적인 학습이 필요한 경우
        - 문서 분류에서 `[CLS]`를 활용한 학습이 이미 진행된 경우

---

**4️⃣ 실제 적용 시 고려할 점**

1. BERT를 사용한다면 `torch.mean()` 방식을 추천
    - `[CLS]` 토큰을 그대로 사용하는 것보다 문장 전체를 평균내는 것이 더 안정적일 가능성이 높음
2. ModernBERT에서는 특성 추출 방식을 고려해야 함
    - pooling layer가 없으므로 `pooler_output`을 기대하면 안되며, `[CLS]` 또는 문장 전체 평균을 활용해야 함
    - 일부 태스크에서 pooling layer가 필요할 경우, ModernBERT 위에 추가적인 Layer를 쌓아야 할 수도 있음
3. 연산량이 중요한 경우 ModernBERT가 더 유리
    - ModernBERT는 BERT보다 약 10~15% 적은 연산량
    - 따라서, 실시간 처리 또는 edge computing 환경에서는 ModernBERT가 더 적합할 가능성이 높음
