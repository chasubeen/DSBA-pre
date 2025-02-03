# **📺 CV 사전학습**


## **Overview**
1. **주제:** 이미지 데이터 학습을 위한 코드 및 결과물 정리  
   - 실험을 위한 모델 코드와 학습 및 평가 코드 작성  
   - 실험 결과 시각화 및 분석  
2. **학습 모델:**  
   - ResNet50  
   - ViT-S/16(Vision Transformer)
3. **실험 항목:**  
   1. ResNet50(w/o pre-trained weights)
   2. ViT-S/16(w/o pre-trained weights)
   3. ResNet50(w/ pre-trained on `ImageNet-1k`)
   4. ViT-S/16(w/ pre-trained on `ImageNet-1k`)
4. **실험 결과:**  
   - 실험 결과를 **Table** 및 **Figure**로 시각화  
   - 모델 성능 비교 및 분석  


## **Directory Structure**
```
CV
├── data/                     # 데이터 파일
│   ├── train_data.npy        # 학습 데이터(이미지)
│   ├── train_target.npy      # 학습 데이터 레이블
│   ├── test_data.npy         # 테스트 데이터(이미지)
│   └── test_target.npy       # 테스트 데이터 레이블
├── config.py                 # 전체 실험 설정 관리(하이퍼파라미터, 경로 등)
├── data.py                   # 데이터 로딩, 전처리 및 DataLoader 정의
├── model/                    # 모델 정의 폴더
│   ├── resnet50.py           # ResNet50 모델 정의(from scratch)
│   └── vit.py                # Vision Transformer(ViT) 모델 정의(timm 라이브러리 활용)
├── metrics.py                # 평가 지표 정의(정확도, Top-k error, 정밀도, 재현율, F1-score)
├── train_eval.py             # 모델 학습/평가 함수 정의
├── main.py                   # 실험 실행을 위한 메인 스크립트
├── model_checkpoints         # 학습된 모델 파일(.pth) 저장
|   ├── resnet50_scratch.pth
|   ├── resnet50_pretrained.pth
|   ├── vit-s_scratch.pth             
|   └── vit-s_pretrained.pth                  
└── results/                  # 실험 결과 시각화 및 분석 폴더
    ├── results.py            # 결과 테이블 생성 및 시각화 코드
    ├── table.csv             # 실험 결과 테이블 저장
    └── figures/              # 실험 결과 그래프 저장
```


## **Descriptions**
### **Configuration(`config.py`)**  
> 모든 실험 설정을 통합 관리할 수 있도록 구조화된 설정 파일
- 데이터 경로, 모델 저장 경로 등 실험 환경 설정 관리
- 학습 관련 주요 설정(optimizer, loss function, learning rate, epochs 등) 정의
- `experiments` 리스트를 통해 여러 실험을 자동화 
- 실험의 재현성을 위해 시드 값(`seed`) 설정 

### **Data(`data/` & `data.py`)**  
> 데이터 로딩 및 전처리 관리
- **`data/` 폴더**  
  - `train_data.npy`, `train_target.npy`: 학습 데이터와 레이블  
  - `test_data.npy`, `test_target.npy`: 테스트 데이터와 레이블  
  - 대용량 데이터의 빠른 로딩을 위해 NumPy 형식으로 저장  
- **`data.py`**
    - `.npy` 형식의 학습/테스트 데이터 로드 및 전처리
    - PyTorch의 `Dataset` 및 `DataLoader` 클래스를 활용하여 데이터 배치 처리
        - 학습과 평가용 데이터 분할 및 랜덤 시드 고정으로 일관된 데이터셋 사용 보장
    - 모델 아키텍처에 맞춰 데이터 전처리 방식 조정
        - ViT의 patch size를 맞추기 위해 **224 x 224**로 입력 이미지 크기 조정(resizing)
        - pre-trained ViT 이미지 프로세서의 mean, std로 정규화 수행
- **데이터 메타정보 확인**
  <details>
    <summary>
        <mark>1️⃣ 데이터 확인용 코드</mark>
    </summary>
    
    ```python
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from collections import Counter
    from data import train_data, train_target, test_data, train_dataset, train_loader
    from config import cfg
    
    # 결과 저장 폴더 설정
    results_dir = os.path.join(cfg.experiment.results_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    
    ### 1) 원본 데이터 정보 확인 (전처리 전)
    print("\n=== 원본 데이터 정보 확인 ===")
    print(f"Train Data 건수: {len(train_data)}")
    print(f"Test Data 건수: {len(test_data)}")
    
    # 고유 클래스 개수 및 분포 확인
    class_counts = Counter(train_target)
    print(f"\n총 클래스 개수: {len(class_counts)}")
    print(f"클래스별 샘플 개수: {class_counts}")
    
    # 원본 데이터 픽셀 값 통계 확인
    train_data_numpy = train_data.astype(np.float32) / 255.0  # 원본 데이터 사용 (0~1로 정규화)
    train_mean = np.mean(train_data_numpy, axis=(0, 1, 2))  # [H, W, C]
    train_std = np.std(train_data_numpy, axis=(0, 1, 2))
    
    print(f"\n원본 데이터 Mean (R, G, B): {train_mean}")
    print(f"원본 데이터 Std  (R, G, B): {train_std}")
    
    # 원본 데이터 샘플 출력
    print("\n원본 데이터 샘플 확인:")
    sample_img = train_data[0]
    sample_label = train_target[0]
    print(f"Sample Image Shape: {sample_img.shape} (H, W, C)")
    print(f"Sample Label: {sample_label}")
    
    ### 2) 원본 데이터 시각화
    print("\n=== 원본 데이터 시각화 ===")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    seen_classes = set()
    for i in range(len(train_data)):
        img = train_data[i]  # 원본 이미지 사용
        label = train_target[i]
    
        if label not in seen_classes:
            axes[label].imshow(img.astype(np.uint8))  # uint8 타입으로 변환 후 출력
            axes[label].axis("off")
            axes[label].set_title(f"Class {label}")
            seen_classes.add(label)
    
        if len(seen_classes) >= 10:  # 모든 클래스 확인되면 종료
            break
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "class_samples.png"))
    plt.show()
    print(f"\n클래스별 원본 샘플 이미지를 {os.path.join(results_dir, 'class_samples.png')}에 저장했습니다!")
    
    ### 3) 데이터로더 적용 후 데이터 정보 확인 (ResNet & ViT 각각 실행)
    for model_type in ["resnet", "vit"]:
        print(f"\n=== 데이터로더 적용 후 데이터 정보 확인 ({model_type.upper()}) ===")
        
        # ✅ 모델 타입 적용
        train_dataset.set_model_type(model_type)
    
        processed_sample = train_dataset[0]  # DataLoader 없이 직접 접근
    
        if isinstance(processed_sample, dict):  # ViT의 경우
            img_sample = processed_sample["pixel_values"]
            label_sample = processed_sample["labels"]
        else:  # ResNet의 경우
            img_sample, label_sample = processed_sample
    
        print(f"DataLoader 적용 후 Sample Image Shape: {img_sample.shape} (C, H, W)")
        print(f"DataLoader 적용 후 Sample Label: {label_sample}")
    
        # 데이터 배치 확인
        batch = next(iter(train_loader))
    
        if isinstance(batch, dict):  # ViT의 경우
            batch_images = batch["pixel_values"]
            batch_labels = batch["labels"]
        else:  # ResNet의 경우
            batch_images, batch_labels = batch
    
        print(f"\nBatch Image Shape: {batch_images.shape}  (Batch, C, H, W)")
        print(f"Batch Label Shape: {batch_labels.shape}")
    ```
    
    </details>
    
    <details>
    <summary>
        <mark>2️⃣ 데이터 확인 결과</mark>
    </summary>  
    === 원본 데이터 시각화 ===  
    <p align="left">    
    <img src="https://github.com/chasubeen/DSBA-pre/blob/main/CV/results/class_samples.png" width=70%>      
    </p>  
        
    ```
    === 원본 데이터 정보 확인 ===  
    Train Data 건수: 20431  
    Test Data 건수: 10000  
    
    총 클래스 개수: 10  
    클래스별 샘플 개수: Counter({0: 5000, 1: 3871, 2: 2997, 3: 2320, 4: 1796, 5: 1391, 6: 1077, 7: 834, 8: 645, 9: 500})  
    
    원본 데이터 Mean (R, G, B): [0.48963538 0.48287246 0.45271188]  
    원본 데이터 Std  (R, G, B): [0.24317575 0.23959668 0.26040143]  
    
    원본 데이터 샘플 확인:  
    Sample Image Shape: (32, 32, 3) (H, W, C)  
    Sample Label: 0  

    
    === 데이터로더 적용 후 데이터 정보 확인(RESNET) ===  
    DataLoader 적용 후 Sample Image Shape: torch.Size([3, 224, 224]) (C, H, W)  
    DataLoader 적용 후 Sample Label: 0  
    
    Batch Image Shape: torch.Size([512, 3, 224, 224])  (Batch, C, H, W)  
    Batch Label Shape: torch.Size([512])  

    
    === 데이터로더 적용 후 데이터 정보 확인(VIT) ===  
    DataLoader 적용 후 Sample Image Shape: torch.Size([3, 224, 224]) (C, H, W)  
    DataLoader 적용 후 Sample Label: 0  
    
    Batch Image Shape: torch.Size([512, 3, 224, 224])  (Batch, C, H, W)  
    Batch Label Shape: torch.Size([512])  
    ```
    
    </details>

### **Model(`model/`)**  
> 모델 정의 및 사전학습 가중치 적용
- **`resnet50.py`**  
  - ResNet50 모델 구조 정의
      - PyTorch로 직접 구현  
  - 사전학습된 가중치 적용 가능(ImageNet-1k)
  - CIFAR-10 데이터셋에 맞게 FC Layer 수정
      - 출력 클래스 수를 10개로 조정

- **`vit.py`**  
  - `timm` 라이브러리를 활용하여 ViT-S/16 모델 로드  
  - 사전학습된 가중치 적용 가능 (ImageNet-1k)
  - CIFAR-10 데이터셋에 맞게 FC Layer 수정
      - 출력 클래스 수를 10개로 조정

### **Metrics(`metrics.py`)**  
> 모델 성능 평가 지표 정의
- 모델의 성능을 측정하기 위한 다양한 평가 지표 정의  
- 정확도(Accuracy), Top-5 Error, 정밀도(Precision), 재현율(Recall), F1 Score 제공
  - 정확도: 전체 샘플 중 정확히 예측한 샘플의 비율
  - Top-5 Error: 모델의 예측 결과 중 상위 5개의 확률이 높은 클래스에 정답이 포함되지 않은 비율
  - 정밀도: 양성(positive)으로 예측한 것 중 실제로 양성인 비율
  - 재현율: 실제 양성인 것 중에서 얼마나 잘 찾아냈는가
  - F1 Score: 정밀도와 재현율의 조화 평균($\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}​$)
- 실험 결과의 신뢰성 확보를 위해 평균 및 표준편차 계산 지원  
- 다중 클래스 분류 문제에 적합하도록 설계  

### **Training & Evaluation(`train_eval.py`)**  
> 모델 학습 및 평가 로직 관리
- **`trainer()`:**  
  - 모델의 학습 루프 정의
      - Forward → Loss 계산 → Backpropagation → Optimizer 업데이트
  - 학습 데이터셋을 통해 손실 함수 및 정확도 계산  

- **`evaluator()`:**  
  - 모델 평가 루프 정의
    - Gradient 비활성화로 메모리 효율성 증가
  - 테스트 데이터셋에 대한 성능 평가 및 주요 지표 산출  

### **Main Script(`main.py`)**  
> 실험 실행을 위한 메인 스크립트
- `config.py`에 정의된 여러 실험을 자동화하여 순차적으로 실행  
- 모델 학습 → 평가 → 결과 저장의 전체 워크플로우 관리
- 실험 로그를 파일로 저장 후 결과 분석 시 활용  
- 학습된 모델을 자동으로 저장
  - Pre-trained 여부 및 모델 이름으로 구분

### **Results(`results/`)**  
> 실험 결과 시각화 및 테이블 생성
- **`results.py`**  
  - `experiment_results.txt` 파일을 파싱하여 실험 결과 데이터프레임으로 변환  
  - 각 실험에 대한 `평균 ± 표준편차` 계산  
  - Table 및 그래프 시각화(Lineplot, Barplot 등) 자동 생성  
- **`table.csv`**  
  - 실험 결과 요약 테이블 저장  
  - Accuracy, Top-5 Error, Precision, Recall, F1 Score를 `평균 ± 표준편차` 형식으로 저장  
- **`figures/` 폴더**  
  - Loss 곡선, 모델별 성능 비교 Barplot 등 시각화 결과 저장
  - 모델 이름 및 Pre-trained 여부에 따라 색상 구분  


## **How to Run**
1. **환경 설정:**  
   ```bash
   pip install -r requirements.txt
   ```
   
2. **모델 학습 및 평가:**  
   ```bash
   python main.py
   ```

3. **결과 시각화:**  
   ```bash
   python results/results.py
   ```

4. **결과 확인:**  
   - **테이블:** `results/table.csv`  
   - **그래프:** `results/figures/` 폴더에 저장  


## **실험 결과(Sample)**
- **Metrics:**
  
    | Model    | Pretrained | Accuracy (%)      | Top-5 Error (%)    | Precision        | Recall           | F1 Score         |
    |----------|------------|-------------------|--------------------|------------------|------------------|------------------|
    | ResNet50 | False      | 63.31 ± 9.87      | 3.86 ± 2.29        | 0.70 ± 0.10      | 0.63 ± 0.10      | 0.63 ± 0.11      |
    | ResNet50 | True       | 92.79 ± 1.24      | 0.21 ± 0.07        | 0.94 ± 0.01      | 0.93 ± 0.01      | 0.93 ± 0.01      |
    | ViT-S    | False      | 51.83 ± 8.72      | 6.93 ± 3.26        | 0.58 ± 0.12      | 0.52 ± 0.09      | 0.50 ± 0.11      |
    | ViT-S    | True       | 95.73 ± 1.03      | 0.10 ± 0.04        | 0.96 ± 0.01      | 0.96 ± 0.01      | 0.96 ± 0.01      |

- **Figures:**
  [figures](https://github.com/chasubeen/DSBA-pre/tree/main/CV/results/figures)
