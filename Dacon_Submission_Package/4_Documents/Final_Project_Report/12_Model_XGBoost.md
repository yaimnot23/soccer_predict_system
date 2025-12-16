# 12. 모델 1: XGBoost (Extreme Gradient Boosting)

## 12.1 모델 선정 이유
XGBoost는 Kaggle 등 정형 데이터 경진대회에서 표준으로 사용되는 모델입니다. 
*   **안정성**: 결측치 처리 및 이상치에 강건(Robust)합니다.
*   **속도**: `tree_method='hist'` 옵션을 통해 대용량 데이터도 빠르게 학습합니다.
*   **성능**: Gradient Boosting 알고리즘을 효율적으로 구현하여 높은 예측 정확도를 보장합니다.

## 12.2 하이퍼파라미터 설정
*   `n_estimators=1000` (조기 종료 적용 시 더 클 수 있음)
*   `learning_rate=0.05`: 세밀한 학습을 위해 낮게 설정.
*   `max_depth=6`: 과적합을 막기 위해 깊이를 제한.
*   `subsample=0.8`, `colsample_bytree=0.8`: 피처와 데이터 샘플링을 통해 일반화 성능 향상.
*   `objective='reg:squarederror'`: 회귀 문제(MSE 최소화) 최적화.

## 12.3 학습 과정
`end_x`와 `end_y`를 각각 예측하는 **Single Target Regression** 모델 2개를 생성했습니다.
(Multi-output Regression보다 각각 최적화하는 것이 미세하게 성능이 좋은 경향이 있음)
