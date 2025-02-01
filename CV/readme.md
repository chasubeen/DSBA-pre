# **📺 CV 사전학습**

## **Overview**
1. 주제: 이미지 데이터 학습을 위한 코드 및 결과물 정리
    - 실험을 위한 모델 코드와 학습 및 평가 코드 작성
    - 실험 결과 작성
2. 학습 모델
    - ResNet50
    - ViT-S/16
3. 실험 항목
    1. ResNet50 w/o pre-trained weights
    2. ViT-S/16 w/o pre-trained weights
    3. ResNet50 w/ pre-trained on ImageNet 1k
    4. ViT-S/16 w/ pre-trained on ImageNet 1k
4. 실험 결과
    - 결과로 보일 수 있는 Table, Figure 모두 작성

## **Directories**
```
CV
├── data/                  # 데이터 파일
    ├── train_data.npy
    ├── train_target.npy
    ├── test_data.npy
    ├── test_target.npy
├── config.py              # 설정 관리
├── data.py                # 데이터 다운로드 및 Dataset, DataLoader 정의
├── model/
    ├── resnet50.py        # ResNet50(from scratch)
    ├── vit.py             # ViT(from timm)
├── metrics.py             # 평가 지표 코드
├── train_eval.py          # 학습 및 평가 코드
├── main.py                # 실행 스크립트
└── results/
    ├── results.py         # 테이블 및 그래프 시각화
    ├── tables/            # 실험 결과 테이블 저장
    └── figures/           # 실험 결과 시각화 저장
```
