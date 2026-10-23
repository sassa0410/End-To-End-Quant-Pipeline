import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def build_features_and_target():
    df = pd.read_parquet('data/raw/crypto_portfolio_closes.parquet')
    
    returns = np.log(df / df.shift(1))

    all_data = []
    FORECAST_HORIZON_MINUTES = 60
    
    logging.info(f"Building features and target for {df.shape[0]} rows and {df.shape[1]} columns...")
    for coin in returns.columns:
        temp_df = pd.DataFrame({'Return': returns[coin]})
        
        #Example of volatility feature: rolling standard deviation of returns over the last 60 minutes
        temp_df['Vol_1h'] = temp_df['Return'].rolling(window=FORECAST_HORIZON_MINUTES).std()
        temp_df['Vol_4h'] = temp_df['Return'].rolling(window=FORECAST_HORIZON_MINUTES * 4).std()
        
        #Example of momentum feature: rolling mean of returns over the last 60 minutes
        temp_df['Mom_1h'] = temp_df['Return'].rolling(window=FORECAST_HORIZON_MINUTES).mean()
        
        #Target: future volatility over the next 60 minutes
        future_vol = temp_df['Return'].rolling(window=FORECAST_HORIZON_MINUTES).std().shift(-FORECAST_HORIZON_MINUTES)
        temp_df['Future_Vol_1h'] = future_vol
        
        temp_df['Coin'] = coin
        
        all_data.append(temp_df)
        
    final_df = pd.concat(all_data)
    final_df.dropna(inplace=True)
    
    os.makedirs('data/processed', exist_ok=True)
    saved_path = 'data/processed/ml_features_.parquet'
    final_df.to_parquet(saved_path)
    logging.info(f"Features for Machine Learning saved successfully in: {saved_path}\n")
    print(f"Sample data:\n{final_df[['Coin', 'Vol_1h', 'Vol_4h', 'Future_Vol_1h']].head()}")
    
if __name__ == "__main__":
    build_features_and_target()