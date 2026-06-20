import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from flaml import AutoML

# 1. Load Data & Target Engineering
df = pd.read_csv('food_wastage_data.csv')
df['Required_Food'] = df['Quantity of Food'] - df['Wastage Food Amount']

features = ['Type of Food', 'Number of Guests', 'Event Type', 'Storage Conditions', 
            'Purchase History', 'Seasonality', 'Preparation Method', 'Geographical Location', 'Pricing']
categorical_features = ['Type of Food', 'Event Type', 'Storage Conditions', 'Purchase History', 
                        'Seasonality', 'Preparation Method', 'Geographical Location', 'Pricing']

# Explicitly call .copy() to completely avoid the SettingWithCopyWarning
X = df[features].copy()
y = df['Required_Food']

# Ensure categorical columns are correctly cast as string/object types
for col in categorical_features:
    X[col] = X[col].astype(str)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize FLAML AutoML
automl = AutoML()

settings = {
    "time_budget": 60,              # Optimization runtime in seconds
    "metric": 'rmse',               # Optimization objective
    "task": 'regression',           # Task type
    "estimator_list": ['catboost'], # Use FLAML's native CatBoost estimator
    "fit_kwargs_by_estimator": {
        "catboost": {
            "verbose": 0            # Suppress CatBoost inner training loop logs
        }
    },
    "log_file_name": 'flaml_catboost.log',
    "seed": 42
}

# 3. Run Search 
print("Starting native FLAML optimization for CatBoost...")
automl.fit(X_train=X_train, y_train=y_train, **settings)

# 4. Output Optimal Results
print("\n--- FLAML Tuning Optimization Summary ---")
print(f"Best Found Hyperparameters: {automl.best_config}")
print(f"Best Validation Loss (RMSE): {automl.best_loss:.4f}")

# 5. Evaluate on Unseen Holdout Test Data
best_predictions = automl.predict(X_test)
print(f"Test Set Final RMSE : {root_mean_squared_error(y_test, best_predictions):.4f}")
print(f"Test Set Final R2   : {r2_score(y_test, best_predictions):.4f}")