# 13. 모델 2: LightGBM (Light Gradient Boosting Machine)

## 13.1 모델 선정 이유
LightGBM은 XGBoost와 유사하지만 다른 트리 성장 방식을 가집니다.
*   **Leaf-wise Growth (리프 중심 성장)**: 트리의 균형을 맞추지 않고 최대 손실 값을 가지는 리프 노드를 지속적으로 분할합니다. 이는 더 깊고 복잡한 패턴을 학습하는 데 유리합니다.
*   **속도**: 매우 빠르며 메모리 사용량이 적습니다.
*   **범주형 변수 처리**: 카테고리형 변수를 자동 분할하는 기능이 우수합니다.

## 13.2 하이퍼파라미터 설정
*   `num_leaves=31`: `max_depth`보다 리프 노드 수로 복잡도를 제어합니다.
*   `learning_rate=0.05`
*   `n_estimators=500`
*   `metric='rmse'`

## 13.3 앙상블 관점에서의 역할
XGBoost(Level-wise)와 LightGBM(Leaf-wise)은 서로 다른 형태의 오차를 가집니다. 두 모델을 결합하면 서로의 약점을 보완해주는 효과가 있어 앙상블 필수 요소로 선택했습니다.
