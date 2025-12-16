# 14. 모델 3: LSTM 개념 및 도입 배경 (LSTM Concept)

## 14.1 왜 딥러닝인가?
트리 모델(XGB, LGBM)은 강력하지만, 데이터의 **"순서(Order)"**를 이해하지 못합니다.
축구 경기는 흐름(Flow)의 싸움입니다.
*   "A → B → C" 로 이어지는 패스웤에서, C는 A와 B의 위치 및 속도에 영향을 받습니다.
*   트리 모델은 이를 Feature Engineering(Lag features)으로 흉내낼 뿐, 본질적인 시퀀스를 이해하진 못합니다.

## 14.2 LSTM (Long Short-Term Memory)
RNN(Recurrent Neural Network)의 단점인 기울기 소실(Vanishing Gradient) 문제를 해결한 모델입니다.
긴 시퀀스 데이터에서도 장기 의존성(Long-term dependency)을 학습할 수 있어, 에피소드 길이가 긴 공격 전개 과정 예측에 적합합니다.

## 14.3 본 프로젝트의 접근
각 에피소드를 하나의 문장(Sentence)처럼 취급하고, 각 이벤트(패스, 드리블)를 단어(Word)처럼 취급하여,
**"지금까지의 이야기(전개)를 들려줄 테니, 그 다음 단어(좌표)를 맞춰봐"**라는 방식으로 학습시켰습니다.
