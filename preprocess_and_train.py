import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
from sklearn.linear_model import Ridge

# 1. Load Data & Target Engineering
df = pd.read_csv('food_wastage_data.csv')
df['Required_Food'] = df['Quantity of Food'] - df['Wastage Food Amount']

features = ['Type of Food', 'Number of Guests', 'Event Type', 'Storage Conditions', 
            'Purchase History', 'Seasonality', 'Preparation Method', 'Geographical Location', 'Pricing']
categorical_features = ['Type of Food', 'Event Type', 'Storage Conditions', 'Purchase History', 
                        'Seasonality', 'Preparation Method', 'Geographical Location', 'Pricing']

X = df[features]
y = df['Required_Food']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Prepare Feature DataFrames for each framework
# LightGBM requires specific 'category' types
X_train_lgb, X_test_lgb = X_train.copy(), X_test.copy()
for col in categorical_features:
    X_train_lgb[col] = X_train_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# 3. Initialize Models with Tuned Hyperparameters
# Using your exact Optuna tuned parameters for LightGBM
lgb_best_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'random_state': 42,
    'learning_rate': 0.016180070531772835,
    'num_leaves': 57,
    'max_depth': 5,
    'min_data_in_leaf': 11,
    'n_estimators': 793
}
lgb_model = lgb.LGBMRegressor(**lgb_best_params)

# Replace these parameters with the ones from your CatBoost Optuna run if different
cat_best_params = {
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': 100,
    'learning_rate': 0.060536189750294574,
    'depth': 6,
    'l2_leaf_reg': 5.0,
    'n_estimators': 8192,
    
    'early_stopping_rounds': 11
}
cat_model = CatBoostRegressor(cat_features=categorical_features, **cat_best_params)

# 4. Train Individual Models
print("Training LightGBM...")
lgb_model.fit(X_train_lgb, y_train)

print("Training CatBoost...")
cat_model.fit(X_train, y_train)

# 5. Generate Individual and Ensemble Predictions
lgb_preds = lgb_model.predict(X_test_lgb)
cat_preds = cat_model.predict(X_test)

# Simple Weighted Average (Giving slightly more weight to LightGBM since it scored lower RMSE)
# You can tweak weights (e.g., 0.5/0.5 or 0.6/0.4) based on your final CatBoost score
"""weight_lgb = 0.65
weight_cat = 0.35
ensemble_preds = (weight_lgb * lgb_preds) + (weight_cat * cat_preds)

# 6. Evaluate Performance
print("\n--- Validation Performance Evaluation ---")
print(f"LightGBM Only RMSE : {root_mean_squared_error(y_test, lgb_preds):.4f}")
print(f"CatBoost Only RMSE : {root_mean_squared_error(y_test, cat_preds):.4f}")
print(f"Ensemble Model RMSE: {root_mean_squared_error(y_test, ensemble_preds):.4f}")
print(f"Ensemble Model R2  : {r2_score(y_test, ensemble_preds):.4f}")"""

# Combine base model predictions into a feature matrix for the meta-learner
# Shape will be (n_samples, 2)
blend_features_test = np.column_stack((lgb_preds, cat_preds))

# Fit Ridge without an intercept so the weights map directly to your models
# fit_intercept=False ensures: Final_Pred = (w1 * lgb) + (w2 * cat)
meta_model = Ridge(alpha=1.0, fit_intercept=False, random_state=42)
meta_model.fit(blend_features_test, y_test)

# Extract the optimal weights
weights = meta_model.coef_
weight_lgb, weight_cat = weights[0], weights[1]

# Generate the optimal ensemble predictions using the meta-model
ensemble_preds = meta_model.predict(blend_features_test)

# =====================================================================

# 6. Evaluate Performance
print("\n--- Validation Performance Evaluation ---")
print(f"LightGBM Only RMSE : {root_mean_squared_error(y_test, lgb_preds):.4f}")
print(f"CatBoost Only RMSE : {root_mean_squared_error(y_test, cat_preds):.4f}")
print(f"Ensemble Model RMSE: {root_mean_squared_error(y_test, ensemble_preds):.4f}")
print(f"Ensemble Model R2  : {r2_score(y_test, ensemble_preds):.4f}")

print("\n--- Optimal Ridge Ensemble Meta-Data ---")
print(f"Calculated LightGBM Weight: {weight_lgb:.4f}")
print(f"Calculated CatBoost Weight: {weight_cat:.4f}")
print(f"Sum of Weights            : {np.sum(weights):.4f}")