from preprocess_and_train import lgb_model, cat_model, categorical_features
import joblib
import numpy as np
class UnifiedEnsemblePredictor:
    def __init__(self, trained_lgb, trained_cat, cat_cols, w_lgb=0.5, w_cat=0.5, rmse_err=21.0):
        self.lgb_model = trained_lgb
        self.cat_model = trained_cat
        self.categorical_features = cat_cols
        self.w_lgb = w_lgb
        self.w_cat = w_cat
        self.safety_buffer = rmse_err * 0.5 # Safety cushion factor to avoid under-catering

    def predict(self, input_df):
        # 1. Format data for LightGBM
        df_lgb = input_df.copy()
        for col in self.categorical_features:
            df_lgb[col] = df_lgb[col].astype('category')
            
        # 2. Get separate predictions
        pred_lgb = self.lgb_model.predict(df_lgb)
        pred_cat = self.cat_model.predict(input_df)
        
        # 3. Blended calculation
        blended_prediction = (self.w_lgb * pred_lgb) + (self.w_cat * pred_cat)
        
        # 4. Return finalized inventory quantity with safety buffer applied
        return np.ceil(blended_prediction + self.safety_buffer)

# Instantiate the ensemble pipeline
production_ensemble = UnifiedEnsemblePredictor(
    trained_lgb=lgb_model, 
    trained_cat=cat_model, 
    cat_cols=categorical_features
)

# Export to disk
joblib.dump(production_ensemble, 'food_wastage_ensemble.pkl')
print("\nProduction Ensemble saved successfully as 'food_wastage_ensemble.pkl'")