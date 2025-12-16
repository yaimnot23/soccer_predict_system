
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# 1. Setup Paths
# Script is in 2_Modeling
# Output is in ../3_Output
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '../3_Output')

files = {
    'xgb': 'submission_xgb_optuna.csv',
    'lgbm': 'submission_lgbm_optuna.csv',
    'cat': 'submission_cat_optuna.csv',
    'lstm': 'submission_lstm.csv'
}

preds = {}

print("Loading Submission Files...")
for model_name, filename in files.items():
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        preds[model_name] = df.set_index('game_episode')
        print(f" [{model_name}] Loaded: {df.shape}")
    else:
        print(f" [WARNING] File Not Found: {filename}")

if not preds:
    print("No predictions found!")
    exit(1)

first_model = list(preds.keys())[0]
common_index = preds[first_model].index
print(f"\nTotal Predictions: {len(common_index)}")

# 2. Weighted Blending
weights = {
    'xgb': 0.25,
    'lgbm': 0.25,
    'cat': 0.25,
    'lstm': 0.25
}

final_x = np.zeros(len(common_index))
final_y = np.zeros(len(common_index))
total_weight = 0

for model_name, w in weights.items():
    if model_name in preds:
        print(f" Blending {model_name} (weight: {w})")
        df = preds[model_name].loc[common_index]
        final_x += df['end_x'].values * w
        final_y += df['end_y'].values * w
        total_weight += w
    else:
        print(f" Skipping {model_name} (Missing)")

if total_weight > 0:
    final_x /= total_weight
    final_y /= total_weight

final_submission = pd.DataFrame({
    'game_episode': common_index,
    'end_x': final_x,
    'end_y': final_y
})

# Clipping
final_submission['end_x'] = np.clip(final_submission['end_x'], 0, 105)
final_submission['end_y'] = np.clip(final_submission['end_y'], 0, 68)

save_path = os.path.join(OUTPUT_DIR, 'final_submission_ensemble_lstm.csv')
final_submission.to_csv(save_path, index=False)

print(f"\n✅ Final Ensemble Saved to: {save_path}")
