# 11. 특성 선택 및 누수 방지 (Feature Selection & Anti-Leakage)

## 11.1 과적합 및 누수 유발 변수 제거
성능을 높이기 위해 무분별하게 변수를 추가하다 보면 대회 규정을 위반하거나 과적합(Overfitting)될 수 있습니다.

### 제거된 주요 변수 (`Dropped Columns`)
1.  **`score_difference` (점수 차)**: 
    *   **이유**: 테스트 시점에 알 수 없는 정보일 가능성이 높으며, 승패 컨텍스트에 지나치게 의존하게 만듦.
2.  **`result_name` (결과)**:
    *   **이유**: 'Fail', 'Success', 'Goal' 등의 결과는 이벤트가 다 끝난 후 기록되는 사후 정보일 수 있어, 입력 피처로 쓰기엔 부적절합니다. (단, 직전 이벤트의 결과는 사용 가능할 수 있으나 안전을 위해 현재 행의 결과는 제외)
3.  **`game_total_goals`**:
    *   **이유**: 명백한 미래 정보(Data Leakage).

## 11.2 최종 Feature Set
결과적으로 모델 학습에는 다음 변수들이 사용되었습니다.
*   **Numerical**: `time_seconds`, `start_x`, `start_y`, `avg_pass_dist`, `avg_pass_angle`, `pass_count` ...
*   **Categorical**: `player_id`, `type_name` (Encoded)
