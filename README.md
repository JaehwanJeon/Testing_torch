# Hysteresis 예측을 위한 딥러닝 모델
## 파일 구조
```
.
├── BilinearTesting_main.py, BoucWenTesting_main.py, BWBNTesting_main.py, IMKTesting_main.py, ROTesting_main.py
│   ├── 각 모델 테스트를 위한 메인 파일
│   ├── 각 파일마다 EQ_generation.py, Preprocessing.py, Training.py, ResultAnalysis.py를 import하여 돌림
│   ├── 각각 generate_EQ, preprocess, train, analyze_result를 True로 설정하여 실행
│   └── 딥러닝 모델의 parameter 또한 이 파일에서 변경 가능
│
├── EQ_generation.py
│   └── EQ 생성 함수
├── Preprocessing.py
│   └── 데이터 전처리 함수
├── Training.py
│   └── 모델 학습 함수
├── ResultAnalysis.py
│   └── 결과 분석 함수
│
└── backend.py
    └── 각종 백엔드 함수 존재 (ex: 손실함수, 제안한 LSTM 모델 등)
```

## 사용 방법
1. generate_EQ, preprocess, train, analyze_result를 True 또는 False로 설정하여 실행
2. preprocess 함수와 train 함수 사이에서 딥러닝 모델의 parameter를 변경 가능 (ex. nn_size = 64, alpha = 0.2, window_size = 1000 등)
