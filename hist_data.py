import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
sns.set_style('darkgrid')

def generate_synthetic_data(n_days=30, n_minutes_per_day=390):
    # Generate timestamps
    base_date = datetime.now()
    timestamps = []
    for day in range(n_days):
        current_date = base_date - timedelta(days=day)
        for minute in range(n_minutes_per_day):
            timestamps.append(current_date + timedelta(minutes=minute))
    timestamps.reverse()
    
    # Generate price data
    n_samples = len(timestamps)
    base_price = 1000
    volatility = 0.02
    
    # Generate random walk prices
    prices = np.random.normal(0, volatility, n_samples).cumsum()
    prices = base_price * (1 + prices)
    
    # Generate OHLCV data
    data = []
    for i in range(0, n_samples, n_minutes_per_day):
        day_prices = prices[i:i+n_minutes_per_day]
        day_volume = np.random.normal(1000000, 100000, n_minutes_per_day)
        day_volume = np.maximum(day_volume, 100000)
        
        for j in range(n_minutes_per_day):
            if i + j < n_samples:
                open_price = day_prices[j] * (1 + np.random.normal(0, 0.001))
                high_price = open_price * (1 + abs(np.random.normal(0, 0.002)))
                low_price = open_price * (1 - abs(np.random.normal(0, 0.002)))
                close_price = (high_price + low_price) / 2
                
                data.append([
                    timestamps[i + j],
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    day_volume[j],
                    0  # OI (Open Interest) - not used in this example
                ])
    
    return pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])

def add_technical_indicators(df):
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Add Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Add RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Add Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    # Add volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Trading volume indicators
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    return df

def prepare_data(df, look_back=60):
    # Drop rows with NaN values
    df = df.dropna()
    
    # Select features for modeling
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'returns', 
                      'SMA_20', 'EMA_20', 'RSI', 'BB_upper', 'BB_lower', 
                      'volatility', 'volume_ratio']
    
    # Scale the features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])
    
    return pd.DataFrame(scaled_data, columns=feature_columns), scaler

# Generate synthetic data
print("Generating synthetic market data...")
hist_data = generate_synthetic_data(n_days=30, n_minutes_per_day=390)

# Add technical indicators
hist_data = add_technical_indicators(hist_data)

# Prepare data for modeling
processed_data, scaler = prepare_data(hist_data)

# Save processed data
hist_data.to_csv('market_data_raw.csv', index=False)
processed_data.to_csv('market_data_processed.csv', index=False)

print("Data processing completed. Shape of processed data:", processed_data.shape)
print("\nFeatures available for modeling:", list(processed_data.columns))

# Plot some basic statistics
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(hist_data['timestamp'], hist_data['close'])
plt.title('Price Over Time')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
plt.plot(hist_data['timestamp'], hist_data['RSI'])
plt.title('RSI Over Time')
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
plt.plot(hist_data['timestamp'], hist_data['volume'])
plt.title('Volume Over Time')
plt.xticks(rotation=45)

plt.subplot(2, 2, 4)
plt.plot(hist_data['timestamp'], hist_data['volatility'])
plt.title('Volatility Over Time')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('market_data_analysis.png')