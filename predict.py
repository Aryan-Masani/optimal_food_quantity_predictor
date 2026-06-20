import pandas as pd
import joblib
import numpy as np




 
class UnifiedEnsemblePredictor:
    def __init__(self, trained_lgb, trained_cat, cat_cols, w_lgb=0.8, w_cat=0.2, rmse_err=21.0):
        
        self.lgb_model = trained_lgb
        self.cat_model = trained_cat
        self.categorical_features = cat_cols
        self.w_lgb = w_lgb
        self.w_cat = w_cat
        

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
        return np.ceil(blended_prediction)
        

# Load pipeline
model = joblib.load('food_wastage_ensemble.pkl')

# Mock upcoming event details
new_event = pd.DataFrame([{
    'Type of Food': 'Meat',
    'Number of Guests': 350,
    'Event Type': 'Wedding',
    'Storage Conditions': 'Refrigerated',
    'Purchase History': 'Regular',
    'Seasonality': 'Winter',
    'Preparation Method': 'Sit-down Dinner',
    'Geographical Location': 'Urban',
    'Pricing': 'Low'
}])

predicted_quantity = model.predict(new_event)
print(f"Optimal Food Quantity to Prepare: {predicted_quantity[0] * 0.45} kg")