# 6. EDA 2: 선수별 성향 분석 (Player Analysis)

## 6.1 Player Persona의 중요성
"같은 위치에서 공을 잡더라도, **누가(Who)** 잡았느냐에 따라 결과는 달라진다."
데이터 분석 결과, `player_id`는 타겟 변수(`end_x`, `end_y`)를 설명하는 가장 강력한 변수 중 하나였습니다.

## 6.2 선수별 클러스터링
선수들의 평균 패스 거리(`avg_pass_dist`)와 각도(`avg_pass_angle`)를 기준으로 K-Means 클러스터링을 시도해 볼 수 있습니다. (시각화 분석 결과)
1.  **Safety First Group (수비수)**: 평균 패스 거리가 짧고, 횡패스 비율이 높음.
2.  **Playmaker Group (미드필더)**: 패스 횟수가 압도적으로 많고, 전방 패스 비율이 높음.
3.  **Long Ball Group (골키퍼/센터백)**: 평균 패스 거리가 매우 길고(`>30m`), 정확도는 상대적으로 떨어짐.

## 6.3 Player ID Handling
`player_id`는 범주형 변수이지만, 카디널리티(고유 값의 개수)가 수백 명에 달합니다. 이를 윈-핫 인코딩(One-Hot Encoding)하기보다는:
1.  **Label Encoding**: 트리 모델이 분기(Split)하기 좋도록 변환.
2.  **Target Encoding (Feature Engineering)**: 해당 선수의 과거 평균 좌표값 등을 피처로 추가하는 방식. (본 프로젝트에서는 Data Leakage 우려로 'Statistics' 방식만 채택)
