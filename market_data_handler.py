import asyncio
import json
import ssl
import websockets
import requests
import pandas as pd
from datetime import datetime
from google.protobuf.json_format import MessageToDict
import MarketDataFeedV3_pb2 as pb
import numpy as np

class MarketDataHandler:
    def __init__(self):
        self.data_buffer = []
        self.current_data = None
        self.is_connected = False
        self.callbacks = []
        
    def get_market_data_feed_authorize_v3(self):
        """Get authorization for market data feed."""
        access_token = 'eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI3WUIzRTkiLCJqdGkiOiI2N2Q2MDE2ZWQwNGViMzEwMWI3NTc0NTIiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaWF0IjoxNzQyMDc4MzE4LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NDIxNjI0MDB9.kCes7hdXuBAI06zsMuVykjx1Kod5aZXnPrXtbqxxbJA'
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        url = 'https://api.upstox.com/v3/feed/market-data-feed/authorize'
        api_response = requests.get(url=url, headers=headers)
        return api_response.json()

    def decode_protobuf(self, buffer):
        """Decode protobuf message."""
        feed_response = pb.FeedResponse()
        feed_response.ParseFromString(buffer)
        return feed_response

    def process_market_data(self, data_dict):
        """Process and format market data."""
        try:
            # Import here to avoid circular import
            from options_trade import add_technical_indicators
            
            # Extract relevant data from the protobuf message
            if 'feed' in data_dict:
                feed = data_dict['feed']
                if 'ltp' in feed:
                    # Convert timestamp to datetime
                    timestamp = datetime.fromtimestamp(feed.get('timestamp', 0))
                    
                    # Create data dictionary with all required fields
                    data = {
                        'timestamp': timestamp,
                        'close': float(feed['ltp']),
                        'open': float(feed.get('open', feed['ltp'])),
                        'high': float(feed.get('high', feed['ltp'])),
                        'low': float(feed.get('low', feed['ltp'])),
                        'volume': float(feed.get('volume', 0))
                    }
                    
                    # Add to buffer (keep last 100 data points)
                    self.data_buffer.append(data)
                    if len(self.data_buffer) > 100:
                        self.data_buffer.pop(0)
                    
                    # If we have enough data for technical indicators
                    if len(self.data_buffer) >= 50:
                        # Create DataFrame with the new data
                        df = pd.DataFrame(self.data_buffer)
                        df.set_index('timestamp', inplace=True)
                        
                        # Add technical indicators
                        df = add_technical_indicators(df)
                        
                        # Update current data with the latest row
                        self.current_data = df.iloc[-1].copy()
                        self.current_data.name = timestamp  # Set the name attribute
                        
                        # Notify callbacks
                        for callback in self.callbacks:
                            callback(self.current_data)
                    else:
                        print(f"Collecting data: {len(self.data_buffer)}/50 points")
                        
        except Exception as e:
            print(f"Error processing market data: {e}")
            print(f"Data dictionary: {data_dict}")
            import traceback
            traceback.print_exc()

    def add_callback(self, callback):
        """Add a callback function to be called when new data arrives."""
        self.callbacks.append(callback)

    async def start_websocket(self):
        """Start WebSocket connection and data processing."""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        try:
            response = self.get_market_data_feed_authorize_v3()
            print("Authorization response:", response)
            
            # Check if the response has the expected structure
            websocket_uri = None
            if "data" in response and "authorized_redirect_uri" in response["data"]:
                websocket_uri = response["data"]["authorized_redirect_uri"]
            elif "authorized_redirect_uri" in response:
                websocket_uri = response["authorized_redirect_uri"]
            
            if not websocket_uri:
                print("Error: Unable to get WebSocket URI from response")
                print("Response:", response)
                return
                
            print(f"Connecting to {websocket_uri}")
            async with websockets.connect(websocket_uri, ssl=ssl_context) as websocket:
                print('WebSocket connection established')
                self.is_connected = True

                await asyncio.sleep(1)

                # Subscribe to market data
                data = {
                    "guid": "someguid",
                    "method": "sub",
                    "data": {
                        "mode": "full",
                        "instrumentKeys": ["NSE_INDEX|Nifty Bank", "NSE_INDEX|Nifty 50"]
                    }
                }

                binary_data = json.dumps(data).encode('utf-8')
                await websocket.send(binary_data)

                while self.is_connected:
                    try:
                        message = await websocket.recv()
                        decoded_data = self.decode_protobuf(message)
                        data_dict = MessageToDict(decoded_data)
                        self.process_market_data(data_dict)
                    except Exception as e:
                        print(f"Error in WebSocket loop: {e}")
                        break
        except Exception as e:
            print(f"Error in WebSocket connection: {e}")
            import traceback
            traceback.print_exc()

    def stop(self):
        """Stop the WebSocket connection."""
        self.is_connected = False

    def get_current_data(self):
        """Get the current market data."""
        return self.current_data

    def get_data_buffer(self):
        """Get the data buffer as a DataFrame."""
        return pd.DataFrame(self.data_buffer)

    async def simulate_market_data(self):
        """Simulate market data for testing."""
        print("Simulating market data...")
        
        # Import here to avoid circular import
        from options_trade import add_technical_indicators
        
        # Start price
        price = 100.0
        volume = 1000
        
        # Simulate data
        while self.is_connected:
            try:
                # Generate random price movement
                price_change = np.random.normal(0, 0.1)
                price = max(0.1, price * (1 + price_change))
                
                # Calculate high, low, open
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = low + (high - low) * np.random.random()
                
                # Random volume
                volume = max(100, volume * (1 + np.random.normal(0, 0.1)))
                
                # Create timestamp
                timestamp = datetime.now()
                
                # Create data dictionary
                data = {
                    'timestamp': timestamp,
                    'close': float(price),
                    'open': float(open_price),
                    'high': float(high),
                    'low': float(low),
                    'volume': float(volume)
                }
                
                # Add to buffer
                self.data_buffer.append(data)
                if len(self.data_buffer) > 100:
                    self.data_buffer.pop(0)
                
                # If we have enough data for technical indicators
                if len(self.data_buffer) >= 50:
                    # Create DataFrame
                    df = pd.DataFrame(self.data_buffer)
                    df.set_index('timestamp', inplace=True)
                    
                    # Add technical indicators
                    df = add_technical_indicators(df)
                    
                    # Update current data
                    self.current_data = df.iloc[-1].copy()
                    self.current_data.name = timestamp
                    
                    # Notify callbacks
                    for callback in self.callbacks:
                        callback(self.current_data)
                else:
                    print(f"Collecting data: {len(self.data_buffer)}/50 points")
                
                # Wait a second
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error in simulation loop: {e}")
                import traceback
                traceback.print_exc()
                break 