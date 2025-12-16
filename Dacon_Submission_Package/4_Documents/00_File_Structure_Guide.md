# 📂 파일 구조 설명 가이드 (File Structure Guide)

이 문서는 `Dacon_Submission_Package` 내의 모든 폴더와 파일의 역할 및 내용을 설명합니다.

---

## 📦 Root Directory (`Dacon_Submission_Package/`)

제출 패키지의 최상위 폴더입니다.

- **`inference.py`**:

  - **역할**: 최종 추론(Inference)을 수행하는 파이썬 스크립트.
  - **내용**: 학습된 모델 가중치를 불러오고, 테스트 데이터를 전처리하여 최종 예측 파일(`final_submission.csv`)을 생성합니다.
  - **실행**: `python inference.py`

- **`requirements.txt`**:
  - **역할**: 프로젝트 실행에 필요한 파이썬 라이브러리 및 버전 목록.
  - **내용**: `pandas`, `xgboost`, `tensorflow` 등 필수 패키지 명시.

---

## 📂 1_Preprocessing/

데이터 전처리 및 특성 공학(Feature Engineering) 관련 파일 모음입니다.

- **`feature_engineering_advanced.ipynb`**:
  - **역할**: 고급 특성 추출 로직(Game State, Player Persona 등)을 개발하고 테스트한 노트북.
- **`feature_engineering_merge.ipynb`**:
  - **역할**: 추출된 특성을 원본 데이터(`train.csv`, `test.csv`)에 병합하여 최종 학습용 데이터(`train_final.csv`)를 만드는 노트북.
- **`train_final.csv` / `test_final.csv`**:
  - **역할**: 전처리가 완료된 학습 및 테스트 데이터 파일. (모델링에 바로 사용됨)
- **`player_features.csv`**:
  - **역할**: `train.csv`에서 추출한 선수별 고유 통계(평균 패스거리 등) 데이터.

---

## 📂 2_Modeling/

모델 학습 및 검증 코드가 포함된 폴더입니다.

### 📂 LSTM/

- **`model_05_lstm.ipynb`**:
  - **역할**: 딥러닝(LSTM) 모델 개발 및 실험 노트북.
- **`run_lstm.py`**:
  - **역할**: LSTM 모델을 학습하고 가중치(`lstm_model.keras`)를 저장하는 실행 스크립트.

### 📂 Tree_Models/

- **`model_01_xgboost.ipynb`** 등:
  - **역할**: 초기 트리 모델 실험 노트북들. (참고용)

### 📄 실행 스크립트

- **`run_tree_models.py`**:
  - **역할**: XGBoost와 LightGBM 모델을 학습하고, 각 Fold별 가중치 파일(`.json`, `.txt`)을 저장하는 스크립트.
- **`run_ensemble.py`**:
  - **역할**: 저장된 예측 파일들(`submission_*.csv`)을 가중 평균하여 앙상블하는 스크립트. (학습 단계에서의 앙상블용)

---

## 📂 3_Output/

모델 학습 결과물 및 예측 파일이 저장되는 폴더입니다.

- **`final_submission_ensemble_lstm.csv`**:
  - **역할**: 최종 제출용 예측 파일 (XGB + LGBM + LSTM 앙상블 결과).
- **`lstm_model.keras`**:
  - **역할**: 학습된 LSTM 모델 가중치 파일.
- **`xgb_*.json` / `lgbm_*.txt`**:
  - **역할**: Fold별로 학습된 트리 모델 가중치 파일들.

---

## 📂 4_Documents/

프로젝트 관련 문서 모음입니다.

- **`00_File_Structure_Guide.md`**: 현재 보고 계신 파일 구조 설명서.
- **`01_Project_Walkthrough.md`**: 프로젝트 수행 과정 및 결과 요약 리포트.
- **`02_Implementation_Plan.md`**: 구현 계획 및 규칙 준수 검증 리포트.
- **`03_Task_Checklist.md`**: 세부 작업 완료 체크리스트.

---
