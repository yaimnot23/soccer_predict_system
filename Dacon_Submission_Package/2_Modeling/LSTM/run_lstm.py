
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import gc

# GPU check
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Fixed Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESS_DIR = os.path.join(BASE_DIR, '../../1_Preprocessing')
OPEN_DIR = os.path.join(BASE_DIR, '../../../open_track1')

# 1. Load Train
print("Loading Train Data...")
train_df = pd.read_csv(os.path.join(PREPROCESS_DIR, 'train_final.csv'))

# 2. Load Test
print("Loading Test Data...")
test_index = pd.read_csv(os.path.join(OPEN_DIR, 'test.csv'))
player_stats = pd.read_csv(os.path.join(PREPROCESS_DIR, 'player_features.csv'))

def load_full_test_data(test_index, player_stats):
    test_list = []
    # Use tqdm but careful with prints
    for idx, row in tqdm(test_index.iterrows(), total=len(test_index), desc="Loading Test Sequences"):
        file_path = os.path.join(OPEN_DIR, row['path'])
        df = pd.read_csv(file_path)
        df = pd.merge(df, player_stats, on='player_id', how='left')
        df['game_episode'] = row['game_episode']
        test_list.append(df)
    return pd.concat(test_list, ignore_index=True)

test_df = load_full_test_data(test_index, player_stats)

fill_values = {
    'avg_pass_dist': player_stats['avg_pass_dist'].mean(),
    'avg_pass_angle': player_stats['avg_pass_angle'].mean(),
    'pass_count': 0
}
test_df.fillna(fill_values, inplace=True)
train_df.fillna(fill_values, inplace=True)

print(f"Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")

# 3. Sequence Generation
NUMERIC_COLS = ['time_seconds', 'start_x', 'start_y', 'avg_pass_dist', 'avg_pass_angle']
CAT_COLS = ['player_id', 'type_name']
TARGET_COLS = ['end_x', 'end_y']

print("Scaling...")
scaler = StandardScaler()
train_df[NUMERIC_COLS] = scaler.fit_transform(train_df[NUMERIC_COLS])
test_df[NUMERIC_COLS] = scaler.transform(test_df[NUMERIC_COLS])

train_df['end_x'] = train_df['end_x'] / 105.0
train_df['end_y'] = train_df['end_y'] / 68.0

print("Encoding...")
for col in CAT_COLS:
    le = LabelEncoder()
    all_vals = pd.concat([train_df[col], test_df[col]]).astype(str).unique()
    le.fit(all_vals)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

MAX_LEN = 20

def create_sequences(df, is_train=True):
    grouped = df.groupby('game_episode')
    seq_numeric = []
    seq_cat1 = []
    seq_cat2 = []
    targets = []
    ids = []
    
    for name, group in tqdm(grouped, desc="Creating Sequences"):
        num_vals = group[NUMERIC_COLS].values
        cat1_vals = group['player_id'].values
        cat2_vals = group['type_name'].values
        
        seq_numeric.append(num_vals)
        seq_cat1.append(cat1_vals)
        seq_cat2.append(cat2_vals)
        ids.append(name)
        
        if is_train:
            target = group.iloc[-1][TARGET_COLS].values
            targets.append(target)
            
    X_num = pad_sequences(seq_numeric, maxlen=MAX_LEN, padding='pre', dtype='float32')
    X_cat1 = pad_sequences(seq_cat1, maxlen=MAX_LEN, padding='pre', dtype='int32')
    X_cat2 = pad_sequences(seq_cat2, maxlen=MAX_LEN, padding='pre', dtype='int32')
    
    if is_train:
        y = np.array(targets, dtype='float32')
        return [X_num, X_cat1, X_cat2], y, ids
    else:
        return [X_num, X_cat1, X_cat2], ids

print("Creating Train Sequences...")
X_train, y_train, train_ids = create_sequences(train_df, is_train=True)
print("Creating Test Sequences...")
X_test, test_ids = create_sequences(test_df, is_train=False)

# 4. Build Model
def build_model(num_len, cat1_vocab, cat2_vocab, emb_dim=16):
    input_num = Input(shape=(None, num_len), name='numeric_in')
    input_cat1 = Input(shape=(None,), name='player_in')
    input_cat2 = Input(shape=(None,), name='type_in')
    
    emb_cat1 = Embedding(input_dim=cat1_vocab, output_dim=emb_dim, mask_zero=True)(input_cat1)
    emb_cat2 = Embedding(input_dim=cat2_vocab, output_dim=8, mask_zero=True)(input_cat2)
    
    concat = Concatenate()([input_num, emb_cat1, emb_cat2])
    
    x = LSTM(64, return_sequences=True)(concat)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    output = Dense(2, activation='linear')(x)
    
    model = Model(inputs=[input_num, input_cat1, input_cat2], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

vocab_player = train_df['player_id'].max() + 1
vocab_type = train_df['type_name'].max() + 1
num_features = X_train[0].shape[2]

model = build_model(num_features, vocab_player, vocab_type)
print("Model Built.")

# 5. Train
print("Training...")
BATCH_SIZE = 64
EPOCHS = 10
# Use 1 epoch just to test if it works, or 10.
# I'll stick to 10 but maybe user will stop me if slow.

# Simple validation split
val_cnt = int(len(X_train[0]) * 0.2)
X_train_fold = [x[:-val_cnt] for x in X_train]
y_train_fold = y_train[:-val_cnt]
X_val_fold = [x[-val_cnt:] for x in X_train]
y_val_fold = y_train[-val_cnt:]

checkpoint = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train_fold, y_train_fold,
    validation_data=(X_val_fold, y_val_fold),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint],
    verbose=1
)

# SAVE MODEL
# Adjust path: ../../3_Output because we are in 2_Modeling/LSTM
model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../3_Output', 'lstm_model.keras')
model.save(model_save_path)
print(f"Saved LSTM Model: {model_save_path}")

# 6. Predict
print("Predicting...")
preds = model.predict(X_test, batch_size=BATCH_SIZE)

preds[:, 0] = preds[:, 0] * 105.0
preds[:, 1] = preds[:, 1] * 68.0
preds[:, 0] = np.clip(preds[:, 0], 0, 105)
preds[:, 1] = np.clip(preds[:, 1], 0, 68)

submission = pd.DataFrame({
    'game_episode': test_ids,
    'end_x': preds[:, 0],
    'end_y': preds[:, 1]
})

sample_sub = pd.read_csv('./open_track1/sample_submission.csv')
final_sub = pd.merge(sample_sub[['game_episode']], submission, on='game_episode', how='left')

save_path = 'submission_lstm.csv'
final_sub.to_csv(save_path, index=False)
print(f"Saved Submission: {save_path}")
