# 3. 데이터 명세 및 구조 (Data Specification)

## 3.1 데이터셋 개요
제공된 데이터는 크게 학습용(`train.csv`), 테스트용(`test.csv`), 경기 정보(`match_info.csv`)로 구성됩니다.

### 3.1.1 Train Data (`train.csv`)
*   **규모**: 약 60만 건 이상의 이벤트 로그.
*   **컬럼(Columns)**:
    *   `game_id`: 경기 식별자.
    *   `game_episode`: 에피소드 식별자 (독립된 시퀀스 단위).
    *   `time_seconds`: 경기 시작 후 경과 시간.
    *   `player_id`: 이벤트를 발생시킨 선수 ID.
    *   `team_id`: 소속 팀 ID.
    *   `type_name`: 이벤트 종류 (Pass, Shot, Dribble 등).
    *   `start_x`, `start_y`: 시작 좌표.
    *   `end_x`, `end_y`: **Target Variable (예측 대상)**.

### 3.1.2 Test Data (`test.csv`)
*   **구조**: 각 에피소드의 마지막 이벤트 정보가 주어지지만, `end_x`, `end_y` 값은 비워져 있거나(Submission 양식), 예측해야 할 시점 직전까지의 로그가 제공됩니다.
*   **특이사항**: Test 셋은 여러 개의 파일로 쪼개져 있거나, 특정 시점에서의 스냅샷 형태로 제공됩니다. 본 프로젝트에서는 전처리를 통해 `test_final.csv`로 통합하여 사용합니다.

### 3.1.3 Match Info (`match_info.csv`)
*   각 `game_id`에 대한 메타 데이터 (날짜, 대진, 결과 등)를 포함하나, 예측 자체에는 독립성 규칙 위반 소지가 있어 제한적으로 참조합니다.
