서울시립대 AI 경진대회: 축구 승패 예측 시스템 데이터 핸드북
1. 데이터셋 개요 (Dataset Overview)
본 대회는 축구 경기 로그 데이터를 기반으로 특정 상황(에피소드)의 마지막 이벤트가 도달할 좌표(end_x, end_y)를 예측하는 과제입니다.

경기장 규격: 105 (가로) x 68 (세로) [피파 공식 규격]

좌표계 특이사항:

start_x, end_x: 0 ~ 105

start_y, end_y: 0 ~ 68

공격 방향 통일 (L → R): 실제 경기 전/후반 진영과 관계없이, 데이터는 항상 왼쪽(0)에서 오른쪽(105)으로 공격하는 것을 기준으로 통일되어 있습니다.

2. 파일별 상세 명세 (File Specifications)
2.1. 메타 데이터 (Metadata)
경기 자체에 대한 배경 정보입니다.

파일명: match_info.csv

Key: game_id

주요 컬럼:

season_id / season_name: 시즌 정보 (예: 3669 = 2024시즌)

competition_id / competition_name: 대회 정보 (예: 587 = K League 1)

home_team_id / away_team_id: 홈/원정 팀 ID 및 이름

home_score / away_score: 최종 스코어 (참고용)

2.2. 학습 데이터 (Train Data)
모델 학습을 위한 실제 선수들의 플레이 로그입니다.

파일명: train.csv

Key: game_id로 match_info와 연결

구조: 경기 > 전/후반(period_id) > 에피소드(episode_id) > 액션(action_id)

주요 컬럼:

game_episode: {game_id}_{episode_id} 형식의 고유 ID.

time_seconds: 경기 시작 후 경과 시간.

type_name: 이벤트 종류 (Pass, Shot, Duel, Carry 등).

result_name: 이벤트 결과 (Successful, Unsuccessful, Goal, Off Target 등).

start_x, start_y: 시작 좌표.

end_x, end_y: 종료 좌표 (예측 타겟).

2.3. 테스트 데이터 (Test Data)
제출을 위해 예측을 수행해야 하는 데이터입니다. 인덱스 파일과 개별 로그 파일로 나뉩니다.

인덱스 파일 (test.csv)

역할: 테스트 데이터 파일의 경로를 안내하는 지도.

컬럼: game_id, game_episode, path (개별 파일 경로, 예: ./test/153363/153363_1.csv).

개별 로그 파일 (./test/.../xxxx_x.csv)

구조: train.csv와 동일한 컬럼을 가짐.

🎯 핵심 차이점: 각 파일의 **가장 마지막 행(Row)**에 end_x, end_y 값이 비어있음(NaN 또는 공란).

목표: 이 비어있는 마지막 행의 좌표를 예측하는 것.

2.4. 제출 양식 (Submission)
파일명: sample_submission.csv

컬럼: game_episode, end_x, end_y

내용: 테스트 데이터의 각 에피소드 ID에 대해 예측한 좌표값을 기입.

3. 이벤트 타입 요약 (Event Types)
주요 이벤트와 성공/실패 기준입니다.

패스/크로스 (Pass/Cross): Successful (동료에게 연결), Unsuccessful (실패).

슈팅 (Shot): Goal (득점), On Target (유효슈팅), Off Target (빗나감), Blocked (수비벽).

운반 (Carry): 공을 몰고 이동 (성공/실패 구분 없음, 좌표 이동 발생).

경합 (Duel): Successful (승리), Unsuccessful (패배).

수비 (Tackle, Interception): 공 소유권을 뺏거나 공격을 차단.