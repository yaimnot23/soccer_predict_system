# 15. 모델 3: LSTM 아키텍처 상세 (LSTM Architecture)

## 15.1 네트워크 구조
Keras를 사용하여 다음과 같은 다중 입력(Multi-Input) 모델을 설계했습니다.

1.  **Embedding Layers (범주형 입력)**
    *   `player_id`: (Batch, 20) → 임베딩 차원 16
    *   `type_name`: (Batch, 20) → 임베딩 차원 8
    *   선수와 이벤트 타입의 의미론적 벡터 공간을 학습합니다.

2.  **Dense/Input Layer (수치형 입력)**
    *   `start_x`, `start_y`, `time` 등 Scaled Numerical Data.
    *   (Batch, 20, num_features)

3.  **Concatenation & LSTM**
    *   임베딩 벡터와 수치형 벡터를 결합(Concat).
    *   **LSTM Layer**: 64 Units. 시퀀스의 마지막 타임스텝의 Hidden State를 출력.
    *   **Dropout (0.3)**: 과적합 방지.

4.  **Output Layer**
    *   **Dense(32, activation='relu')**: 비선형 변환.
    *   **Dense(2, activation='linear')**: 최종 예측값 `end_x`, `end_y`.

## 15.2 Loss Function & Optimizer
*   **Loss**: `MSE` (Mean Squared Error). 유클리드 거리 오차를 최소화하는 데 적합.
*   **Optimizer**: `Adam`. 학습률 자동 조절.
*   **Metric**: `MAE` (Mean Absolute Error). 직관적인 거리 오차 확인.
