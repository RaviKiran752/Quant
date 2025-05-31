import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm
from model import BiLSTMAttention
from market_data_handler import MarketDataHandler
import asyncio
import threading
import time
from sklearn.preprocessing import MinMaxScaler

def add_technical_indicators(df):
    """Add technical indicators to the DataFrame."""
    # Calculate EMAs
    df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_medium'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Calculate Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Calculate ATR for volatility
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    return df

def prepare_data(df, sequence_length=10):
    """Prepare data for the model."""
    # Select features for the model
    features = ['close', 'volume', 'rsi', 'macd', 'signal', 'atr', 
                'bb_upper', 'bb_lower', 'ema_fast', 'ema_medium', 'ema_slow']
    
    # Create sequences
    sequences = []
    for i in range(len(df) - sequence_length):
        sequence = df[features].iloc[i:(i + sequence_length)].values
        sequences.append(sequence)
    
    return np.array(sequences)

def predict_next_price(model, data, scaler):
    """Predict the next price using the model."""
    model.eval()
    with torch.no_grad():
        # Ensure data is 3D (batch_size, sequence_length, features)
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        
        # Make prediction
        prediction = model(data)
        return prediction.item()

def generate_trading_signal(current_data, model, scaler, sequence_length=10):
    """Generate trading signal based on current market data and model prediction."""
    try:
        if current_data is None:
            return 'HOLD'
            
        # Get the data buffer
        df = market_handler.get_data_buffer()
        if len(df) < sequence_length:
            return 'HOLD'
        
        # Convert data buffer to DataFrame and add technical indicators
        df = pd.DataFrame(df)
        df.set_index('timestamp', inplace=True)
        df = add_technical_indicators(df)
        
        # Prepare data for prediction
        sequences = prepare_data(df, sequence_length)
        if len(sequences) == 0:
            return 'HOLD'
        
        # Get model prediction
        prediction = predict_next_price(model, torch.FloatTensor(sequences[-1]), scaler)
        
        # Get current indicators
        current_price = current_data['close']
        
        # Check if all required indicators are present
        required_indicators = ['ema_fast', 'ema_medium', 'ema_slow', 'rsi', 'macd', 'signal', 'atr']
        if not all(indicator in current_data for indicator in required_indicators):
            return 'HOLD'
            
        ema_fast = current_data['ema_fast']
        ema_medium = current_data['ema_medium']
        ema_slow = current_data['ema_slow']
        rsi = current_data['rsi']
        macd = current_data['macd']
        signal = current_data['signal']
        atr = current_data['atr']
        
        # Generate signal based on multiple conditions
        signal = 'HOLD'
        
        # Trend conditions
        trend_up = ema_fast > ema_medium > ema_slow
        trend_down = ema_fast < ema_medium < ema_slow
        
        # Momentum conditions
        momentum_up = rsi > 50 and macd > signal
        momentum_down = rsi < 50 and macd < signal
        
        # Volatility condition
        volatility_ok = atr < 0.03
        
        # Price prediction conditions
        price_up = prediction > current_price
        price_down = prediction < current_price
        
        # Generate signals
        if trend_up and momentum_up and volatility_ok and price_up:
            signal = 'BUY_CALL'
        elif trend_down and momentum_down and volatility_ok and price_down:
            signal = 'BUY_PUT'
        
        return signal
        
    except Exception as e:
        print(f"Error in generate_trading_signal: {e}")
        return 'HOLD'

class OptionsTradingEnvironment:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.positions = []
        self.trade_history = []
        self.current_price = None
        self.volatility = 0.2  # 20% annual volatility
        self.risk_free_rate = 0.05  # 5% annual risk-free rate
        self.days_to_expiry = 30  # Default option expiry
        
    def calculate_option_price(self, strike_price, option_type='call'):
        """Calculate option price using Black-Scholes model."""
        if self.current_price is None:
            return 0
            
        S = self.current_price
        K = strike_price
        T = self.days_to_expiry / 365
        r = self.risk_free_rate
        sigma = self.volatility
        
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            
        return price
        
    def execute_option_trade(self, signal, current_data):
        """Execute option trade based on signal."""
        self.current_price = current_data['close']
        
        # Close existing positions if signal is opposite
        for position in self.positions[:]:
            if (signal == 'BUY_CALL' and position['type'] == 'put') or \
               (signal == 'BUY_PUT' and position['type'] == 'call'):
                self.close_position(position, current_data)
        
        # Calculate position size (10% of cash balance)
        position_size = self.cash_balance * 0.1
        
        if signal == 'BUY_CALL':
            # Calculate strike price (at-the-money)
            strike_price = round(self.current_price)
            option_price = self.calculate_option_price(strike_price, 'call')
            
            if option_price > 0:
                quantity = int(position_size / option_price)
                if quantity > 0:
                    cost = quantity * option_price
                    self.cash_balance -= cost
                    self.positions.append({
                        'type': 'call',
                        'strike': strike_price,
                        'quantity': quantity,
                        'entry_price': option_price,
                        'entry_time': current_data.name
                    })
                    self.trade_history.append({
                        'timestamp': current_data.name,
                        'action': 'BUY_CALL',
                        'price': option_price,
                        'quantity': quantity,
                        'cost': cost
                    })
                    
        elif signal == 'BUY_PUT':
            # Calculate strike price (at-the-money)
            strike_price = round(self.current_price)
            option_price = self.calculate_option_price(strike_price, 'put')
            
            if option_price > 0:
                quantity = int(position_size / option_price)
                if quantity > 0:
                    cost = quantity * option_price
                    self.cash_balance -= cost
                    self.positions.append({
                        'type': 'put',
                        'strike': strike_price,
                        'quantity': quantity,
                        'entry_price': option_price,
                        'entry_time': current_data.name
                    })
                    self.trade_history.append({
                        'timestamp': current_data.name,
                        'action': 'BUY_PUT',
                        'price': option_price,
                        'quantity': quantity,
                        'cost': cost
                    })
    
    def close_position(self, position, current_data):
        """Close an option position."""
        self.current_price = current_data['close']
        option_price = self.calculate_option_price(position['strike'], position['type'])
        
        if option_price > 0:
            proceeds = position['quantity'] * option_price
            self.cash_balance += proceeds
            self.trade_history.append({
                'timestamp': current_data.name,
                'action': f"SELL_{position['type'].upper()}",
                'price': option_price,
                'quantity': position['quantity'],
                'proceeds': proceeds
            })
            self.positions.remove(position)
    
    def calculate_position_value(self, position, current_data):
        """Calculate the current value of a position."""
        self.current_price = current_data['close']
        option_price = self.calculate_option_price(position['strike'], position['type'])
        return position['quantity'] * option_price

def handle_market_data(data):
    """Handle incoming market data."""
    global market_handler, model, scaler, env
    
    try:
        if data is None:
            print("Received None data")
            return
            
        # Generate trading signal
        signal = generate_trading_signal(data, model, scaler)
        
        # Execute trade based on signal
        env.execute_option_trade(signal, data)
        
        # Print current state
        print(f"\nTimestamp: {data.name}")
        print(f"Current Price: ${data['close']:.2f}")
        print(f"Signal: {signal}")
        
        # Print indicators if they exist
        if 'ema_fast' in data:
            print(f"EMAs - Fast: ${data['ema_fast']:.2f}, Medium: ${data['ema_medium']:.2f}, Slow: ${data['ema_slow']:.2f}")
        if 'rsi' in data:
            print(f"RSI: {data['rsi']:.2f}")
        if 'macd' in data:
            print(f"MACD: {data['macd']:.2f}")
        if 'atr' in data:
            print(f"Volatility: {data['atr']:.2f}")
            
        print(f"Cash Balance: ${env.cash_balance:.2f}")
        
        # Calculate total equity
        total_equity = env.cash_balance
        for position in env.positions:
            total_equity += env.calculate_position_value(position, data)
        print(f"Total Equity: ${total_equity:.2f}")
        
    except Exception as e:
        print(f"Error in handle_market_data: {e}")
        print(f"Data received: {data}")
        import traceback
        traceback.print_exc()

async def run_options_trading():
    """Run the options trading simulation with real-time data."""
    global market_handler, model, scaler, env
    
    # Initialize components
    market_handler = MarketDataHandler()
    model = BiLSTMAttention(input_dim=11, hidden_dim=100, num_layers=2)
    scaler = MinMaxScaler()
    env = OptionsTradingEnvironment()
    
    # Load trained model if available
    try:
        model.load_state_dict(torch.load('bilstm_attention_model.pth'))
        print("Loaded trained model")
    except:
        print("No trained model found, using untrained model for demonstration")
    
    # Add callback for market data
    market_handler.add_callback(handle_market_data)
    
    # Try to connect to real market data
    try:
        print("Attempting to connect to real market data...")
        connect_task = asyncio.create_task(market_handler.start_websocket())
        # Wait for 10 seconds maximum
        await asyncio.wait_for(connect_task, 10.0)
    except Exception as e:
        print(f"Failed to connect to real market data: {e}")
        print("Falling back to simulated market data")
        
        # Use simulated data instead
        market_handler.is_connected = True
        await market_handler.simulate_market_data()

if __name__ == "__main__":
    # Run the trading simulation
    asyncio.run(run_options_trading()) 