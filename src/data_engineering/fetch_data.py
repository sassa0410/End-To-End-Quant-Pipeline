import pandas as pd
from binance.client import Client
import os
import time

def fetch_top_10_crypto_data():
    # 1. Configuration
    client = Client()
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT', 
               'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'AVAXUSDT', 'UNIUSDT']
    
    interval = Client.KLINE_INTERVAL_1MINUTE
    lookback = "7 days ago UTC" # Selecting a 7-day lookback period for more comprehensive data
    
    os.makedirs('../../data/raw', exist_ok=True)
    
    df_portfolio = pd.DataFrame()

    # 2. Data Fetching
    for symbol in symbols:
        print(f"Downloading data for {symbol}...")
        try:
            klines = client.get_historical_klines(symbol, interval, lookback)
            print(f"  -> Found {len(klines)} candles for {symbol}")
            
            if not klines:
                print(f"  -> No data found for {symbol}. Skipping.")
                continue
            
            colonne = ['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
                       'Close_Time', 'Quote_Asset_Volume', 'Number_of_Trades', 
                       'Taker_Buy_Base', 'Taker_Buy_Quote', 'Ignore']
            
            df_temp = pd.DataFrame(klines, columns=colonne)
            df_temp['Open_Time'] = pd.to_datetime(df_temp['Open_Time'], unit='ms')
            df_temp.set_index('Open_Time', inplace=True)
            
            close_prices = pd.to_numeric(df_temp['Close'])
            
            close_prices.name = symbol
            
            # Uniamo i dati al DataFrame principale
            if df_portfolio.empty:
                df_portfolio = pd.DataFrame(close_prices)
            else:
                # Il join allinea automaticamente i dati basandosi sul timestamp (Open_Time)
                df_portfolio = df_portfolio.join(close_prices, how='outer')
                
            # Pausa di mezzo secondo per essere gentili con i server Binance (Rate Limiting)
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")

    if df_portfolio.empty:
        print("\nERROR: The final dataset is completely empty.")
        return

    # 3. Pulizia dei Dati (Gestione della sporcizia)
    print("\nChecking for missing values (Gaps in time):")
    print(df_portfolio.isnull().sum())
    
    # Riempiamo gli eventuali buchi portando avanti l'ultimo prezzo valido (Forward Fill)
    df_portfolio.ffill(inplace=True)
    # Rimuoviamo le primissime righe se per caso mancano dati iniziali per alcune monete
    df_portfolio.dropna(inplace=True)
    
    # 4. Salvataggio in formato Parquet
    percorso_salvataggio = '../../data/raw/crypto_portfolio_closes.parquet'
    df_portfolio.to_parquet(percorso_salvataggio)
    
    print(f"\nData saved successfully in: {percorso_salvataggio}")
    print(f"Dataset dimensions: {df_portfolio.shape}")
    print(df_portfolio.head())

if __name__ == "__main__":
    fetch_top_10_crypto_data()