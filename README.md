# Santander Customer Transaction Prediction

## Team Roles
- Yousef Syed: Data Pipeline & Preprocessing Features
- Blake Cillis: Baselines & Stats Features
- Kirill Delyukin: Models & Validation

## Structure
- `data/`: Input CSVs (ignored by git)
- `src/`: Python source code
- `notebooks/`: Exploration notebooks

## Setup Instructions
1. Clone repo
2. Unzip `santander_magic_data.zip` into `data/` folder
3. Run `pip install -r requirements.txt`

## Run Models
1. LGMB: python -m src.model_lgbm
2. XGB: python -m src.model_xgb
3. Generate File: python -m src.submission_generator


