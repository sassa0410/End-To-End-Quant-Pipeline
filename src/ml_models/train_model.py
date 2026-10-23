import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def train_and_predict_model():
    # 1. Load Features and Target
    features_path = 'data/processed/ml_features_.parquet'
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found at {features_path}. Please run the feature engineering step first.")
    
    df = pd.read_parquet(features_path)
    features = ['Vol_1h', 'Vol_4h', 'Mom_1h']
    target = 'Future_Vol_1h'
    
    df = df.sort_index()
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    x_train, y_train = train_df[features], train_df[target]
    x_test, y_test = test_df[features], test_df[target]
    
    # 2. Models Training
    models = {
        'Baseline_Ridge' : Ridge(alpha=1.0, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
    }
    
    best_model_name = None
    best_model = None
    best_mae = float('inf')
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_model_name = name
    
    logging.info(f"Best model: {best_model_name} with MAE: {best_mae:.6f}")

    # 3. Save the Best Model
    os.makedirs('src/saved_models', exist_ok=True)
    path_model = 'src/saved_models/best_volatility_model.pkl'
    joblib.dump(best_model, path_model)
    logging.info(f"Best model saved successfully in: {path_model}")

    # 4. Predicting Future Volatility with the Best Model
    logging.info('Predicting future volatility for the test set...')
    latest_data = df.groupby('Coin').tail(1)
    
    live_predictions = {}
    for _, row in latest_data.iterrows():
        coin = row['Coin']
        x_live = row[features].values.reshape(1, -1)
        prediction = best_model.predict(x_live)[0]
        live_predictions[coin] = max(0, prediction)  #Ensure non-negative prediction
        
    preds = pd.DataFrame(list(live_predictions.items()), columns=['Coin', 'Predicted_Vol_1h'])
    preds_path = 'data/processed/live_predictions.csv'
    preds.to_csv(preds_path, index=False)
    logging.info(f"Live predictions saved successfully!\n")
    print(preds.to_string(index=False))

if __name__ == "__main__":
    train_and_predict_model()