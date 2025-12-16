# 구현 계획 - K-리그 승패 예측 (좌표 예측)

## 목표

선수들의 플레이 마지막 패스 도달 좌표(x, y) 예측 정확도(Euclidean Distance 최소화)를 개선합니다.
현재의 트리 기반 앙상블 시스템에 **딥러닝(시계열 모델)**을 추가하고, 축구 경기의 맥락을 반영하는 **고급 특성 공학**을 적용합니다.

## 사용자 검토 필요 사항

> [!IMPORTANT] > **새로운 딥러닝 트랙**: PyTorch/TensorFlow 기반의 LSTM/GRU 모델 추가를 제안합니다. 이는 기존의 정형 데이터 방식과는 다른 데이터 전처리(시퀀스 변환)가 필요합니다.
>
> **계산 리소스**: 딥러닝 학습은 트리 모델보다 시간이 더 소요될 수 있습니다.

## 변경 제안 내용

### 1. 데이터 및 특성 공학 (Data & Feature Engineering)

#### [수정] `data_preprocess/` 파이프라인

- **경기 상태(Game State) 재구성**:
  - `game_id` 내 이벤트를 순회하며 현재 점수(홈 vs 원정)를 계산합니다.
  - `score_difference` (점수 차) 변수 추가 (이고 있거나 지고 있을 때 플레이 스타일이 다름).
- **ID 인코딩**:
  - `player_id`와 `team_id`가 존재하지만 현재 충분히 활용되지 않음.
  - 트리 모델용: **Target Mean Encoding** ("선수별 평균 패스 위치" 등) 적용.
  - 딥러닝용: **ID Embeddings** 준비.

### 2. 딥러닝 모델 (신규)

#### [신규] `model_05_lstm.ipynb`

- **아키텍처**:
  - `player_id`, `team_id`, `action_type`에 대한 임베딩 레이어.
  - 에피소드 내 일련의 행동을 처리하는 LSTM/GRU 레이어.
  - Dense Output Layer -> (x, y) 예측.
- **기대 효과**: 시간을 개별 변수로 보는 트리 모델보다 "궤적"과 "흐름"을 더 잘 포착함.

### 3. 검증 전략 (Validation Strategy)

#### [수정] 모든 학습 노트북

- **GroupKFold**: `game_id`를 기준으로 검증 세트를 분리합니다.
- 무작위 분할 시 같은 경기의 데이터가 학습/검증에 섞여 점수가 부풀려지는(Data Leakage) 현상을 방지합니다.

## 검증 계획

### 자동화 테스트

- **Shape 확인**: 시퀀스 데이터 형상 확인 (Batch, Seq_Len, Features).
- **누수(Leakage) 점검**: 학습 세트와 검증 세트 간 `game_id` 중복 여부 확인.

### 수동 검증

- **리더보드 시뮬레이션**: 로컬 검증 점수(OOF)와 Public 리더보드 점수 추세 비교.

## ⚖️ 대회 규칙 준수 검증 리포트 (Compliance Report)

**[규칙 1] 외부 데이터 사용 금지**

- ✅ **준수함**: `train.csv`, `test.csv` 외 어떠한 외부 파일도 로드하지 않음.
- ✅ `player_stats`는 오직 `train.csv`에서만 추출됨.

**[규칙 2] 게임-에피소드 독립성 (Game-Episode Independence)**

- ✅ **준수함**: `Train Game ID`와 `Test Game ID` 간의 **교집합(Overlap)이 0임**을 확인했습니다.
- ✅ 따라서 `train.csv`에서 학습한 '선수 성향'을 `test`에 적용하는 것은, **"과거의 기억(Train)"을 가지고 "미래의 경기(Test)"를 예측**하는 합법적인 ML 방식입니다.
- ✅ **조치**: 같은 경기 내 다른 에피소드 정보를 사용하는 `score_difference` 변수는 **규칙 위반 소지**가 있어 **삭제**했습니다.

**[규칙 3] 평가 데이터 누수 (Data Leakage)**

- ✅ **준수함**: Test 파일 처리를 `test_index`를 통해 개별 파일 단위로 수행하여, 다른 테스트 파일의 정보를 참조하지 않도록 격리했습니다.
