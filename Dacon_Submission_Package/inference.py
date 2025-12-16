
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings

warnings.filterwarnings('ignore')

# 환경 설정 (Configuration)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 디렉토리 구조 가정: Dacon.../inference.py
# 모델 위치: Dacon.../3_Output
MODEL_DIR = os.path.join(BASE_DIR, '3_Output')
PREPROCESS_DIR = os.path.join(BASE_DIR, '1_Preprocessing')
# 테스트 데이터는 보통 제공되거나 open_track1에서 읽어옵니다.
# 제출을 위해, 테스트 데이터가 알려진 위치에 있거나 인자로 전달된다고 가정합니다.
# 여기서는 PREPROCESS_DIR의 test_final.csv 내용을 확인합니다.

def load_data():
    print("데이터 로딩 중...")
    # 추론은 이상적으로 원본(Raw) 테스트 데이터부터 시작해야 합니다.
    # 하지만 엄격한 규칙 준수 절차의 단순화를 위해, 선수 통계(Player Stats)가 이미 병합된 'test_final.csv'를 사용합니다.
    # 만약 원본 테스트 데이터를 사용해야 한다면, 선수 통계를 다시 병합하는 과정이 필요합니다.
    # 여기서는 'test_final.csv'가 존재한다고 가정합니다.
    test_path = os.path.join(PREPROCESS_DIR, 'test_final.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"테스트 데이터를 찾을 수 없습니다: {test_path}")
    
    test = pd.read_csv(test_path)
    # score_diff 및 점수 관련 변수 제거 (Data Leakage / 규칙 위반 방지)
    drop_cols = ['score_diff', 'current_team_score', 'opp_team_score', 'game_total_goals']
    test = test.drop(columns=[c for c in drop_cols if c in test.columns])
    return test

def preprocess_tree(test_df):
    # 특성 선택 (Features)
    exclude_cols = ['game_id', 'period_id', 'episode_id', 'game_episode', 'end_x', 'end_y', 'is_goal', 'goal_val']
    features = [c for c in test_df.columns if c not in exclude_cols]
    
    # 인코딩 (Encoding)
    # 중요: 인코딩 로직은 학습(Training) 시점과 동일해야 합니다.
    # 원래는 인코더(Encoder) 객체를 저장해서 불러와야 하지만, 편의상 Train 데이터를 로드하여 다시 적합(Fit)합니다.
    # Test 데이터만으로 인코딩하면 범주가 불일치할 위험이 있습니다.
    # 최선책: Train 데이터를 로드하여 전체 범주를 파악합니다.
    train_path = os.path.join(PREPROCESS_DIR, 'train_final.csv')
    train = pd.read_csv(train_path)
    
    cat_cols = ['type_name', 'result_name', 'team_id', 'player_id']
    cat_cols = ['type_name', 'result_name', 'team_id', 'player_id']
    for col in cat_cols:
        if col in train.columns:
            le = LabelEncoder()
            # Train과 Test의 모든 값을 합쳐서 범주를 학습합니다.
            all_vals = pd.concat([train[col], test_df[col]]).astype(str).unique()
            le.fit(all_vals)
            test_df[col] = le.transform(test_df[col].astype(str))
            
    return test_df[features], test_df['game_episode']

def predict_xgb(X_test, game_episodes):
    print("XGBoost 추론 중...")
    preds_final = np.zeros((len(X_test), 2))
    
    for i, target in enumerate(['end_x', 'end_y']):
        # 5-Fold 앙상블 평균
        fold_preds = np.zeros(len(X_test))
        for fold in range(5):
            model_path = os.path.join(MODEL_DIR, f'xgb_{target}_fold{fold}.json')
            if os.path.exists(model_path):
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                fold_preds += model.predict(X_test)
            else:
                print(f"경고: 모델 파일을 찾을 수 없습니다 - {model_path}")
        preds_final[:, i] = fold_preds / 5
        
    return preds_final

def predict_lgbm(X_test, game_episodes):
    print("LightGBM 추론 중...")
    preds_final = np.zeros((len(X_test), 2))
    
    for i, target in enumerate(['end_x', 'end_y']):
        fold_preds = np.zeros(len(X_test))
        for fold in range(5):
            model_path = os.path.join(MODEL_DIR, f'lgbm_{target}_fold{fold}.txt')
            if os.path.exists(model_path):
                model = lgb.Booster(model_file=model_path)
                fold_preds += model.predict(X_test)
            else:
                print(f"경고: 모델 파일을 찾을 수 없습니다 - {model_path}")
        preds_final[:, i] = fold_preds / 5
        
    return preds_final

def predict_lstm(test_df):
    print("Predicting LSTM...")
    # LSTM Preprocessing (Sequence)
    # Re-impl sequence gen logic or import?
    # Better to copy-paste essential logic to match training exactly.
    
    # 1. Load Train for Scaling fit
    train_path = os.path.join(PREPROCESS_DIR, 'train_final.csv')
    train_df = pd.read_csv(train_path)
    
    NUMERIC_COLS = ['time_seconds', 'start_x', 'start_y', 'avg_pass_dist', 'avg_pass_angle']
    CAT_COLS = ['player_id', 'type_name']
    
    scaler = StandardScaler()
    scaler.fit(train_df[NUMERIC_COLS])
    
    test_scaled = test_df.copy()
    test_scaled[NUMERIC_COLS] = scaler.transform(test_scaled[NUMERIC_COLS])
    
    # Encode
    for col in CAT_COLS:
        le = LabelEncoder()
        all_vals = pd.concat([train_df[col], test_scaled[col]]).astype(str).unique()
        le.fit(all_vals)
        test_scaled[col] = le.transform(test_scaled[col].astype(str))
        
    # Seq Gen
    MAX_LEN = 20
    grouped = test_scaled.groupby('game_episode')
    seq_numeric = []
    seq_cat1 = []
    seq_cat2 = []
    ids = []
    
    for name, group in grouped:
        num_vals = group[NUMERIC_COLS].values
        cat1_vals = group['player_id'].values
        cat2_vals = group['type_name'].values
        seq_numeric.append(num_vals)
        seq_cat1.append(cat1_vals)
        seq_cat2.append(cat2_vals)
        ids.append(name)
        
    X_num = pad_sequences(seq_numeric, maxlen=MAX_LEN, padding='pre', dtype='float32')
    X_cat1 = pad_sequences(seq_cat1, maxlen=MAX_LEN, padding='pre', dtype='int32')
    X_cat2 = pad_sequences(seq_cat2, maxlen=MAX_LEN, padding='pre', dtype='int32')
    
    # Load Model
    model_path = os.path.join(MODEL_DIR, 'lstm_model.keras')
    model = tf.keras.models.load_model(model_path)
    
    preds = model.predict([X_num, X_cat1, X_cat2], verbose=0)
    
    # Inverse Scale Target (0~1 to 105/68)
    preds[:, 0] = preds[:, 0] * 105.0
    preds[:, 1] = preds[:, 1] * 68.0
    
    return preds, ids

def main():
    print("추론 시작 (Starting Inference)...")
    try:
        test_df = load_data()
    except Exception as e:
        print(f"데이터 로딩 오류: {e}")
        return

    # 트리 모델 추론 (Tree Inference)
    print("트리 모델(XGB, LGBM) 추론 시작...")
    # 트리 모델용 전처리 (인코딩 포함)
    X_tree, game_episodes = preprocess_tree(test_df.copy())
    
    p_xgb = predict_xgb(X_tree, game_episodes)
    p_lgbm = predict_lgbm(X_tree, game_episodes)
    
    # LSTM 추론 (LSTM Inference)
    # 참고: LSTM은 전체 시퀀스 로그가 필요합니다. 만약 'test_final.csv'(마지막 행만 있음)를 사용한다면 LSTM 실행이 불가능합니다.
    # 이번 제출 패키지에서는 편의상 'test_final.csv'가 트리 모델에 충분하다고 가정하고 진행합니다.
    # 따라서, 외부 의존성을 최소화하고 실행 안전성을 보장하기 위해 엄격한 추론 스크립트에서는 트리 모델 앙상블만 사용합니다.
    # (LSTM 가중치가 있더라도 데이터 로딩 로직이 복잡하므로 여기서는 제외)
    # 우리는 XGB + LGBM (50:50) 블렌딩을 사용하여 결과를 생성합니다.
    
    print("예측값 결합 (Blending Predictions)...")
    final_preds = (p_xgb * 0.5) + (p_lgbm * 0.5)
    
    # 값 보정 (Clipping: 경기장 크기 0~105, 0~68)
    final_preds[:, 0] = np.clip(final_preds[:, 0], 0, 105)
    final_preds[:, 1] = np.clip(final_preds[:, 1], 0, 68)
    
    submission = pd.DataFrame({
        'game_episode': game_episodes,
        'end_x': final_preds[:, 0],
        'end_y': final_preds[:, 1]
    })
    
    save_path = 'final_submission.csv'
    submission.to_csv(save_path, index=False)
    print(f"Inference Complete. Saved to {save_path}")

if __name__ == '__main__':
    main()
