import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna

# Force Optuna to be quiet during trials
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load data
df = pd.read_csv('food_wastage_data.csv')
df['Required_Food'] = df['Quantity of Food'] - df['Wastage Food Amount']

features = ['Type of Food', 'Number of Guests', 'Event Type', 'Storage Conditions', 
            'Purchase History', 'Seasonality', 'Preparation Method', 'Geographical Location', 'Pricing']
categorical_features = ['Type of Food', 'Event Type', 'Storage Conditions', 'Purchase History', 
                        'Seasonality', 'Preparation Method', 'Geographical Location', 'Pricing']

X = df[features]
y = df['Required_Food']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Copy data specifically for LightGBM's category types
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()
for col in categorical_features:
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# Objective function for LightGBM
def objective_lgb(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
        'n_estimators': trial.suggest_int('n_estimators', 100, 800)
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train_lgb, y_train)
    preds = model.predict(X_test_lgb)
    return root_mean_squared_error(y_test, preds)

print("Optimizing LightGBM...")
study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=50)
print(f"Best LightGBM RMSE: {study_lgb.best_value:.4f}")
print(f"Best LightGBM Params: {study_lgb.best_params}\n")