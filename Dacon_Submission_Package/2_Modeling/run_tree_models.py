
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
import gc

warnings.filterwarnings('ignore')

# Paths
# Fixed Paths with Normpath
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
PREPROCESS_DIR = os.path.normpath(os.path.join(BASE_DIR, '../1_Preprocessing'))
OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, '../3_Output'))

print(f"Base Dir: {BASE_DIR}")
print(f"Preprocess Dir: {PREPROCESS_DIR}")
print(f"Output Dir: {OUTPUT_DIR}")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading Data...")
train = pd.read_csv(os.path.join(PREPROCESS_DIR, 'train_final.csv'))
test = pd.read_csv(os.path.join(PREPROCESS_DIR, 'test_final.csv'))

# Preprocessing for Tree Models
# 1. Drop Leakage/Skew Columns
# score_difference: Train has values, Test is 0. Data Leakage/Skew Risk -> Drop.
drop_cols = ['score_diff', 'current_team_score', 'opp_team_score', 'game_total_goals']
train = train.drop(columns=[c for c in drop_cols if c in train.columns])
test = test.drop(columns=[c for c in drop_cols if c in test.columns])

# 2. Features Selection
# Coordinates, Time, Player Stats, etc.
# Exclude non-feature columns
exclude_cols = ['game_id', 'period_id', 'episode_id', 'game_episode', 'end_x', 'end_y', 'is_goal', 'goal_val']
features = [c for c in train.columns if c not in exclude_cols]

# 3. Label Encoding (Categorical)
cat_cols = ['type_name', 'result_name', 'team_id', 'player_id'] # Treat IDs as cat or num? Tree models handle num well, but player_id is nominal.
# For simplicity and speed, LabelEncode large cardinality or let models handle it.
# XGB/LGBM prefer encoded.

print("Encoding Categoricals...")
for col in cat_cols:
    if col in train.columns:
        le = LabelEncoder()
        # Fit on both to catch all categories
        all_vals = pd.concat([train[col], test[col]]).astype(str).unique()
        le.fit(all_vals)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        
        # Explicit convert to category type for LGBM/Cat if needed, but int is fine.

X = train[features]
y = train[['end_x', 'end_y']]
X_test = test[features]

print(f"Features ({len(features)}): {features}")

# Validation Strategy
gkf = GroupKFold(n_splits=5)
groups = train['game_id']

# Models
def train_predict(model_type='xgb'):
    print(f"\nTraining {model_type}...")
    oof_preds = np.zeros(y.shape)
    test_preds = np.zeros((len(test), 2))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        if model_type == 'xgb':
            model = xgb.XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method='hist', # Faster
                device='cpu',
                random_state=42,
                n_jobs=-1
            )
            # MultiOutputReg? No, XGBRegressor supports multi-target naturally? Not really, usually better to fit 2 models or MultiOutput wrapper.
            # But XGB natively supports multi-target in recent versions? Let's use MultiOutput wrapper logic manually or sklearn wrapper.
            # Actually, simplest is separate models for x and y, or sklearn's MultiOutputRegressor.
            # BUT, let's keep it simple: Loop for X and Y targets.
            pass

        # To keep code simple and robust: Fit for end_x, then end_y
        
    # Redesign: Loop X and Y separately
    preds_final = np.zeros((len(test), 2))
    
    for i, target_col in enumerate(['end_x', 'end_y']):
        print(f"  Target: {target_col}")
        y_target = y[target_col]
        test_pred_col = np.zeros(len(test))
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_target, groups)):
            X_tr, y_tr = X.iloc[train_idx], y_target.iloc[train_idx]
            X_v, y_v = X.iloc[val_idx], y_target.iloc[val_idx]
            
            if model_type == 'xgb':
                model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, n_jobs=-1, random_state=42)
                model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
                # Save XGB
                save_name = f'{model_type}_{target_col}_fold{fold}.json'
                model.save_model(os.path.join(OUTPUT_DIR, save_name))
                
            elif model_type == 'lgbm':
                model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, n_jobs=-1, random_state=42, verbose=-1)
                model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], eval_metric='rmse', callbacks=[lgb.early_stopping(50, verbose=False)])
                # Save LGBM
                save_name = f'{model_type}_{target_col}_fold{fold}.txt'
                model.booster_.save_model(os.path.join(OUTPUT_DIR, save_name))

            elif model_type == 'cat':
                model = cb.CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_state=42, verbose=0, allow_writing_files=False)
                model.fit(X_tr, y_tr, eval_set=(X_v, y_v), early_stopping_rounds=50)
            
            test_pred_col += model.predict(X_test) / 5 # 5 folds
            
        preds_final[:, i] = test_pred_col
        
    return preds_final

# Run Training
for m in ['xgb', 'lgbm']:
    preds = train_predict(m)
    
    # Post-process
    preds[:, 0] = np.clip(preds[:, 0], 0, 105)
    preds[:, 1] = np.clip(preds[:, 1], 0, 68)
    
    submission = pd.DataFrame({
        'game_episode': test['game_episode'],
        'end_x': preds[:, 0],
        'end_y': preds[:, 1]
    })
    
    # Save with same name format as before to match ensemble script
    save_path = os.path.join(OUTPUT_DIR, f'submission_{m}_optuna.csv')
    submission.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

print("All Tree Models Updated.")
