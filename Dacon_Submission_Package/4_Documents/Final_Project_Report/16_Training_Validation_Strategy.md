# 16. 학습 및 검증 전략 (Training & Validation)

## 16.1 GroupKFold Cross Validation
모든 모델 학습 시 `GroupKFold`는 선택이 아닌 **필수**였습니다.
*   **이유**:
    *   단순 KFold 시: `Game A`의 전반전은 Train, 후반전은 Valid에 들어갈 수 있음 → 후반전 예측 시 이미 전반전 패턴을 알고 있는 셈(Leakage).
    *   GroupKFold 시: `Game A` 전체가 통째로 Fold 1의 Valid 셋에만 존재. 학습 때는 `Game A`를 전혀 보지 못함.
*   이 방식은 실제 리더보드(Test Set) 환경(완전히 새로운 경기 예측)과 가장 유사한 평가 결과를 제공합니다.

## 16.2 Early Stopping (조기 종료)
*   학습 과정에서 **Validation Score가 더 이상 개선되지 않으면(patience=50)** 학습을 중단하고 최적의 가중치(Best iteration)를 저장/복원했습니다.
*   이는 불필요한 컴퓨팅 자원 낭비를 막고 과적합을 방지하는 핵심 기법입니다.

## 16.3 GPU 활용
*   XGBoost, LightGBM, TensorFlow 모두 GPU 가속을 활용할 수 있도록 환경을 구성했으나,
*   최종 제출 코드(`inference.py`)의 호환성을 위해 CPU 기반 추론이 가능하도록 설계했습니다.
