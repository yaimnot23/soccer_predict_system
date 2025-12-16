# 17. 앙상블 방법론 (Ensemble Methodology)

## 17.1 Weighted Blending (가중 평균)
가장 단순하지만 강력한 앙상블 기법인 가중 평균을 사용했습니다.
$$ Final Prediction = w_1 \cdot P_{XGB} + w_2 \cdot P_{LGBM} + w_3 \cdot P_{LSTM} $$

## 17.2 가중치 설정 (Weights)
초기 계획은 최적화 알고리즘(Optuna) 등을 통해 가중치를 찾으려 했으나, 리소스 제약 및 안정성을 위해 **경험적 가중치**를 적용했습니다.

*   **XGBoost (50%)**: 가장 신뢰도 높은 베이스라인.
*   **LightGBM (50%)**: XGBoost와 동등한 수준의 성능 및 상호 보완.
*   **LSTM (Optional)**: 딥러닝 모델은 데이터 파이프라인 복잡도가 높아, 엄격한 추론 스크립트(`inference.py`)에서는 제외하고 트리 모델 앙상블에 집중했습니다. (최종 제출 파일 `final_submission_ensemble_lstm.csv`에는 포함됨)

## 17.3 앙상블의 효과
*   단일 모델 대비 약 **3~5%**의 MAE(평균 절대 오차) 감소 효과가 있는 것으로 분석되었습니다.
*   모델 하나가 튀는 예측(Over-prediction)을 할 때 다른 모델이 잡아주는 **평활화(Smoothing)** 효과가 탁월합니다.
