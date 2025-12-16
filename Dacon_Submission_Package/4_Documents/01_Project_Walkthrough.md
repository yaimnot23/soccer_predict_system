# K-리그 승패 예측 프로젝트 요약

## 🏆 최종 결과물

- **제출 파일**: `Dacon_Submission_Package/3_Output/final_submission_ensemble_lstm.csv`
- **핵심 기술**: Tree Models (XGB/LGBM/Cat) + **Deep Learning (LSTM)**

## 🛠️ 수행 작업 (Workflow)

### 1. 규칙 준수 및 데이터 보호 (Compliance)

- [x] **외부 데이터 미사용**: 오직 제공된 `train.csv`만 사용했습니다.
- [x] **독립성 검증**: Train과 Test의 `game_id`가 겹치지 않음을 확인하여, **전체 선수 통계(Persona)**를 활용하는 것이 합법적임을 증명했습니다.
- [x] **Data Leakage 방지**: `score_difference` 변수는 과적합 및 리규 위반 위험으로 **제거**했습니다.

### 2. 고급 특성 공학 (Feature Engineering)

- **Player Persona**: 각 선수의 평균 패스 거리, 각도, 빈도를 추출하여 "누가 차는지"에 대한 맥락을 모델에 제공했습니다.
- **Scaling & Encoding**: 좌표 데이터를 0~1로 정규화하고, ID를 임베딩할 수 있도록 인코딩했습니다.

### 3. 딥러닝 모델 (LSTM)

- **Sequence Modeling**: `game_episode` 내의 이벤트 흐름(패스->드리블->패스)을 시계열로 학습했습니다.
- **Embedding Layers**: 선수(Player)와 이벤트 타입(Action)을 벡터로 변환하여 학습 효율을 높였습니다.

### 4. 앙상블 (Ensemble)

- **최종 모델**: **XGBoost + LightGBM + LSTM** (3개 모델 앙상블).
- **CatBoost**: 메모리 제약으로 제외되었습니다.
- 모든 모델은 **엄격한 규칙 준수(Strict Compliance)** 하에 재학습되었습니다 (`score_diff` 제거).

## 📂 폴더 구조 (Dacon_Submission_Package)

- **1_Preprocessing**: 데이터 전처리 코드
- **2_Modeling**: 모델링 코드 (Tree & LSTM)
- **3_Output**: 예측 결과 파일
- **4_Documents**: 기획서 및 메모
