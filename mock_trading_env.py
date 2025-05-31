import asyncio
import json
import ssl
import websockets
import requests
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from google.protobuf.json_format import MessageToDict
import MarketDataFeedV3_pb2 as pb
from sklearn.preprocessing import MinMaxScaler
from model_building import BiLSTMAttention
import joblib

class MockTradingEnvironment:
    def __init__(self, initial_balance=100000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = 0
        self.trades = []
        self.prices = []
        self.predictions = []
        self.performance_history = []
        self.total_profit_loss = 0
        self.last_buy_price = 0
        
        # Load the trained scaler
        self.scaler = joblib.load("scaler.pkl")
        
        # Initialize data buffer
        self.data_buffer = []
        self.seq_length = 10
        
        # Trading parameters
        self.instrument_name = "NIFTY 50"
        self.instrument_type = "INDEX"
        
        # Load the trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BiLSTMAttention(input_dim=6, hidden_dim=64, output_dim=1)
        self.model.load_state_dict(torch.load('bilstm_attention_model.pth', map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
    
    def update_data_buffer(self, price_data):
        """Update the data buffer with new price data"""
        self.data_buffer.append(price_data)
        if len(self.data_buffer) > self.seq_length:
            self.data_buffer.pop(0)
    
    def make_prediction(self):
        """Make prediction using the BiLSTM+Attention model"""
        if len(self.data_buffer) < self.seq_length:
            return None
            
        # Convert data buffer to numpy array and reshape
        data = np.array(self.data_buffer)
        
        # Normalize the data
        data = self.scaler.transform(data)
        
        # Convert to tensor and add batch dimension
        data = torch.FloatTensor(data).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(data)
            prediction = prediction.cpu().numpy()[0][0]
            
        # Create a dummy array for inverse transform
        dummy_array = np.zeros((1, 6))
        dummy_array[0, 3] = prediction  # Put prediction in the 'close' position
        
        # Denormalize prediction
        denormalized = self.scaler.inverse_transform(dummy_array)
        return denormalized[0, 3]  # Return the denormalized close price
    
    def calculate_profit_loss(self, current_price):
        """Calculate profit/loss for current position"""
        if self.positions > 0:
            return (current_price - self.last_buy_price) * self.positions
        return 0
    
    def execute_trade(self, current_price, prediction):
        """Execute trading logic based on predictions"""
        if prediction is None:
            return
            
        # Calculate current profit/loss
        current_pnl = self.calculate_profit_loss(current_price)
            
        # Trading strategy based on model prediction
        if prediction > current_price * 1.001 and self.positions <= 0:  # 0.1% threshold
            # Buy
            shares = self.balance // current_price
            if shares > 0:
                self.positions = shares
                self.balance -= shares * current_price
                self.last_buy_price = current_price
                self.trades.append({
                    'timestamp': datetime.now(),
                    'action': 'BUY',
                    'instrument': self.instrument_name,
                    'type': self.instrument_type,
                    'price': current_price,
                    'shares': shares,
                    'profit_loss': 0
                })
                print(f"\nBUY {self.instrument_name} ({self.instrument_type})")
                print(f"Price: ₹{current_price:.2f}")
                print(f"Quantity: {shares} shares")
                print(f"Total Value: ₹{shares * current_price:.2f}")
                
        elif prediction < current_price * 0.999 and self.positions >= 0:  # 0.1% threshold
            # Sell
            if self.positions > 0:
                self.balance += self.positions * current_price
                profit_loss = (current_price - self.last_buy_price) * self.positions
                self.total_profit_loss += profit_loss
                
                self.trades.append({
                    'timestamp': datetime.now(),
                    'action': 'SELL',
                    'instrument': self.instrument_name,
                    'type': self.instrument_type,
                    'price': current_price,
                    'shares': self.positions,
                    'profit_loss': profit_loss
                })
                
                print(f"\nSELL {self.instrument_name} ({self.instrument_type})")
                print(f"Price: ₹{current_price:.2f}")
                print(f"Quantity: {self.positions} shares")
                print(f"Total Value: ₹{self.positions * current_price:.2f}")
                print(f"Profit/Loss: ₹{profit_loss:.2f}")
                print(f"Total P/L: ₹{self.total_profit_loss:.2f}")
                
                self.positions = 0
    
    def update_performance(self, current_price):
        """Update performance metrics"""
        portfolio_value = self.balance + (self.positions * current_price)
        current_pnl = self.calculate_profit_loss(current_price)
        
        self.performance_history.append({
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'positions': self.positions,
            'cash': self.balance,
            'current_price': current_price,
            'profit_loss': current_pnl,
            'total_profit_loss': self.total_profit_loss
        })
        
        # Print current portfolio status
        print(f"\nPortfolio Status:")
        print(f"Value: ₹{portfolio_value:.2f}")
        print(f"Cash: ₹{self.balance:.2f}")
        print(f"Positions: {self.positions} shares")
        print(f"Current P/L: ₹{current_pnl:.2f}")
        print(f"Total P/L: ₹{self.total_profit_loss:.2f}")
    
    def plot_performance(self):
        """Plot trading performance"""
        df = pd.DataFrame(self.performance_history)
        
        plt.figure(figsize=(15, 12))
        
        # Plot portfolio value
        plt.subplot(3, 1, 1)
        plt.plot(df['timestamp'], df['portfolio_value'])
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Time')
        plt.ylabel('Value (₹)')
        
        # Plot positions
        plt.subplot(3, 1, 2)
        plt.plot(df['timestamp'], df['positions'])
        plt.title('Position Size Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Shares')
        
        # Plot profit/loss
        plt.subplot(3, 1, 3)
        plt.plot(df['timestamp'], df['total_profit_loss'])
        plt.title('Total Profit/Loss Over Time')
        plt.xlabel('Time')
        plt.ylabel('P/L (₹)')
        
        plt.tight_layout()
        plt.savefig('trading_performance.png')
        plt.close()

def get_market_data_feed_authorize_v3():
    """Get authorization for market data feed."""
    access_token = 'eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3WUIzRTkiLCJqdGkiOiI2N2ViN2UxY2JmM2I5ODA1ZjUyYzAyY2IiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaWF0IjoxNzQzNDg2NDkyLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NDM1NDQ4MDB9.nv8VUl8RD3AdTcPIwDqUU8AeSNI3mfCrm-7fhPxkkkA'
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    url = 'https://api.upstox.com/v3/feed/market-data-feed/authorize'
    api_response = requests.get(url=url, headers=headers)
    return api_response.json()

def decode_protobuf(buffer):
    """Decode protobuf message."""
    feed_response = pb.FeedResponse()
    feed_response.ParseFromString(buffer)
    return feed_response

async def fetch_market_data():
    """Fetch market data using WebSocket and process it"""
    env = MockTradingEnvironment()
    
    # Create default SSL context
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # Get market data feed authorization
    response = get_market_data_feed_authorize_v3()
    
    # Connect to the WebSocket with SSL context
    async with websockets.connect(response["data"]["authorized_redirect_uri"], ssl=ssl_context) as websocket:
        print('Connection established')

        await asyncio.sleep(1)

        # Subscribe to Nifty 50 data
        data = {
            "guid": "someguid",
            "method": "sub",
            "data": {
                "mode": "full",
                "instrumentKeys": ["NSE_INDEX|Nifty 50"]
            }
        }

        binary_data = json.dumps(data).encode('utf-8')
        await websocket.send(binary_data)

        while True:
            message = await websocket.recv()
            decoded_data = decode_protobuf(message)
            data_dict = MessageToDict(decoded_data)
            
            # Extract price data
            if 'feeds' in data_dict and 'NSE_INDEX|Nifty 50' in data_dict['feeds']:
                feed = data_dict['feeds']['NSE_INDEX|Nifty 50']
                if 'fullFeed' in feed and 'indexFF' in feed['fullFeed']:
                    price_data = feed['fullFeed']['indexFF']['ltpc']
                    current_price = float(price_data['ltp'])
                    
                    # Get OHLC data
                    ohlc_data = feed['fullFeed']['indexFF']['marketOHLC']['ohlc'][0]  # Get daily OHLC
                    
                    # Update environment with all required features
                    env.update_data_buffer([
                        float(ohlc_data['open']),    # Open price
                        float(ohlc_data['high']),    # High price
                        float(ohlc_data['low']),     # Low price
                        float(price_data['ltp']),    # Current price
                        0,                           # Volume (not available for index)
                        0                            # Open Interest (not available for index)
                    ])
                    
                    # Make prediction and execute trade
                    prediction = env.make_prediction()
                    env.execute_trade(current_price, prediction)
                    env.update_performance(current_price)
                    
                    # Store data for visualization
                    env.prices.append(current_price)
                    if prediction:
                        env.predictions.append(prediction)
                    
                    # Plot performance every 100 data points
                    if len(env.performance_history) % 100 == 0:
                        env.plot_performance()

# Execute the mock trading environment
if __name__ == "__main__":
    asyncio.run(fetch_market_data()) 