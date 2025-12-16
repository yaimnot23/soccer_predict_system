# 8. 데이터 전처리 전략 (Preprocessing Strategy)

## 8.1 데이터 정제 (Data Cleaning)
*   **결측치(Null)**: `train.csv`에는 결측치가 거의 없으나, 일부 이벤트 타입 누락은 `Unknown`으로 대체하거나 제거했습니다.
*   **좌표 스케일링**: 좌표값(0~105, 0~68)은 물리적 거리를 의미하므로, 트리 모델에서는 그대로 사용하되 딥러닝 모델(LSTM)에서는 0~1 사이로 Min-Max Scaling 했습니다.

## 8.2 데이터 분할 (Train/Validation Split)
가장 중요한 것은 **Data Leakage 방지**입니다.
*   **Random Split 금지**: 단순히 `shuffle`하여 나누면, 같은 경기의 인접한 에피소드들이 Train과 Valid에 섞여 들어가 미래 정보를 참조하게 될 위험이 큽니다.
*   **GroupKFold 적용**: `game_id`를 그룹 키로 사용하여, **특정 경기의 모든 데이터는 오직 Train 혹은 Validation 중 한 곳에만 속하게** 했습니다.
    *   `n_splits=5`: 5-Fold 교차 검증을 통해 모델의 일반화 성능을 확보했습니다.

## 8.3 시퀀스 데이터 구성 (For LSTM)
*   트리 모델은 각 행(Raw)을 독립적으로 보지만, LSTM은 "과거 20개 이벤트의 흐름"을 봐야 합니다.
*   `groupby('game_episode')`를 통해 각 에피소드를 리스트로 묶고, `pad_sequences`를 사용하여 길이를 20으로 맞췄습니다. (짧으면 0으로 채움)
