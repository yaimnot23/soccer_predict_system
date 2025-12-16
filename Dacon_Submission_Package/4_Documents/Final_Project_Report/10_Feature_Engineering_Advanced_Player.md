# 10. 특성 공학 2: 선수 고유 특성 (Feature Engineering - Advanced)

본 프로젝트의 **Key Factor**입니다.

## 10.1 Player Statistics (Stats)
`train.csv` 전체를 집계하여 각 `player_id`별 통계를 생성했습니다.
1.  **`avg_pass_dist` (평균 패스 거리)**:
    *   이 값이 크면 롱 패스를 즐기는 선수(예: 기성용 스타일)임을 의미합니다.
2.  **`avg_pass_angle` (평균 패스 각도)**:
    *   이 값이 0에 가까우면 전진 패스 위주, 90도나 -90도에 가까우면 횡패스 위주임을 의미합니다.
3.  **`pass_count` (패스 시도 횟수)**:
    *   데이터의 신뢰도 가중치로 활용됩니다. 패스 횟수가 적은 신인 선수는 통계값의 신뢰도가 낮으므로 모델이 보수적으로 판단하게 돕습니다.

## 10.2 적용 방식
위에서 생성한 `player_stats` 테이블을 `train` 및 `test` 데이터의 `player_id` 기준으로 Left Join 했습니다.
테스트 셋에 처음 보는 선수(New Player)가 있을 경우, 전체 선수의 **평균값(Global Mean)**으로 결측치를 대체하여 에러를 방지했습니다.
