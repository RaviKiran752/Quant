import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from model import BiLSTMAttention, TimeSeriesDataset
from sklearn.preprocessing import MinMaxScaler

class MockTradingEnvironment:
    def __init__(self, initial_capital=100000, position_size=0.1, stock_symbol="RELIANCE"):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_size = position_size  # Fraction of capital to use per trade
        self.position = 0  # Current position (0: no position, 1: long, -1: short)
        self.entry_price = 0
        self.trades = []
        self.equity_curve = []
        self.stock_symbol = stock_symbol
        self.cash_balance = initial_capital
        self.position_value = 0
        self.unrealized_pnl = 0
        
    def calculate_position_size(self, price):
        return (self.current_capital * self.position_size) / price
    
    def get_position_status(self):
        if self.position == 0:
            return "No Position"
        elif self.position > 0:
            return f"Long {self.position:.2f} shares"
        else:
            return f"Short {abs(self.position):.2f} shares"
    
    def get_balance_summary(self, current_price):
        self.position_value = self.position * current_price
        self.unrealized_pnl = self.position_value - (self.position * self.entry_price) if self.position != 0 else 0
        return {
            'cash_balance': self.cash_balance,
            'position_value': self.position_value,
            'unrealized_pnl': self.unrealized_pnl,
            'total_equity': self.current_capital + self.unrealized_pnl
        }
    
    def execute_trade(self, action, price, timestamp):
        balance_before = self.get_balance_summary(price)
        
        if action == 1 and self.position <= 0:  # Buy signal
            if self.position < 0:  # Close short position
                profit = (self.entry_price - price) * abs(self.position)
                self.current_capital += profit
                self.cash_balance += profit
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'close_short',
                    'price': price,
                    'shares': abs(self.position),
                    'profit': profit,
                    'cash_balance': self.cash_balance,
                    'position_value': 0,
                    'total_equity': self.current_capital
                })
            
            # Open long position
            position_size = self.calculate_position_size(price)
            cost = position_size * price
            self.cash_balance -= cost
            self.position = position_size
            self.entry_price = price
            self.trades.append({
                'timestamp': timestamp,
                'action': 'open_long',
                'price': price,
                'shares': position_size,
                'cost': cost,
                'cash_balance': self.cash_balance,
                'position_value': cost,
                'total_equity': self.current_capital
            })
            
        elif action == -1 and self.position >= 0:  # Sell signal
            if self.position > 0:  # Close long position
                profit = (price - self.entry_price) * self.position
                self.current_capital += profit
                self.cash_balance += (self.position * price)
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'close_long',
                    'price': price,
                    'shares': self.position,
                    'profit': profit,
                    'cash_balance': self.cash_balance,
                    'position_value': 0,
                    'total_equity': self.current_capital
                })
            
            # Open short position
            position_size = self.calculate_position_size(price)
            self.position = -position_size
            self.entry_price = price
            self.trades.append({
                'timestamp': timestamp,
                'action': 'open_short',
                'price': price,
                'shares': position_size,
                'cash_balance': self.cash_balance,
                'position_value': -position_size * price,
                'total_equity': self.current_capital
            })
        
        balance_after = self.get_balance_summary(price)
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': balance_after['total_equity'],
            'cash_balance': balance_after['cash_balance'],
            'position_value': balance_after['position_value'],
            'unrealized_pnl': balance_after['unrealized_pnl']
        })
    
    def close_all_positions(self, price, timestamp):
        if self.position > 0:  # Close long position
            profit = (price - self.entry_price) * self.position
            self.current_capital += profit
            self.cash_balance += (self.position * price)
            self.trades.append({
                'timestamp': timestamp,
                'action': 'close_long',
                'price': price,
                'shares': self.position,
                'profit': profit,
                'cash_balance': self.cash_balance,
                'position_value': 0,
                'total_equity': self.current_capital
            })
        elif self.position < 0:  # Close short position
            profit = (self.entry_price - price) * abs(self.position)
            self.current_capital += profit
            self.cash_balance += profit
            self.trades.append({
                'timestamp': timestamp,
                'action': 'close_short',
                'price': price,
                'shares': abs(self.position),
                'profit': profit,
                'cash_balance': self.cash_balance,
                'position_value': 0,
                'total_equity': self.current_capital
            })
        self.position = 0
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.current_capital,
            'cash_balance': self.cash_balance,
            'position_value': 0,
            'unrealized_pnl': 0
        })
    
    def get_performance_metrics(self):
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        winning_trades = trades_df[trades_df['profit'] > 0]
        losing_trades = trades_df[trades_df['profit'] < 0]
        
        # Calculate max drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        rolling_max = equity_df['equity'].expanding().max()
        drawdowns = (equity_df['equity'] - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0.02)
        returns = equity_df['equity'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * (returns.mean() - 0.02) / returns.std()
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades),
            'total_profit': self.current_capital - self.initial_capital,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }

def predict_next_price(model, data, scaler, look_back=60):
    model.eval()
    with torch.no_grad():
        # Prepare input sequence
        sequence = data[-look_back:].values
        sequence = torch.FloatTensor(sequence).unsqueeze(0)
        
        # Make prediction
        prediction = model(sequence)
        return prediction.item()

def run_mock_trading():
    # Load data
    raw_data = pd.read_csv('market_data_raw.csv')
    processed_data = pd.read_csv('market_data_processed.csv')
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMAttention(input_dim=processed_data.shape[1], hidden_dim=50)
    model.load_state_dict(torch.load('bilstm_attention_model.pth'))
    model.to(device)
    
    # Initialize trading environment
    env = MockTradingEnvironment(initial_capital=100000, position_size=0.1, stock_symbol="RELIANCE")
    
    print(f"\nStarting Mock Trading for {env.stock_symbol}")
    print(f"Initial Capital: ${env.initial_capital:,.2f}")
    print(f"Position Size: {env.position_size*100}% of capital per trade")
    
    # Run trading simulation
    look_back = 60
    for i in range(look_back, len(processed_data)):
        # Get current price
        current_price = raw_data.iloc[i]['close']
        current_timestamp = raw_data.iloc[i]['timestamp']
        
        # Make prediction
        prediction = predict_next_price(model, processed_data.iloc[:i], None, look_back)
        
        # Determine trading action based on prediction
        if prediction > 0.001:  # Bullish signal
            action = 1
            signal = "BUY"
        elif prediction < -0.001:  # Bearish signal
            action = -1
            signal = "SELL"
        else:  # Neutral signal
            action = 0
            signal = "HOLD"
        
        # Get current balance and position status
        balance = env.get_balance_summary(current_price)
        position_status = env.get_position_status()
        
        # Print trading information
        print(f"\nTimestamp: {current_timestamp}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Signal: {signal} (Prediction: {prediction:.4f})")
        print(f"Position: {position_status}")
        print(f"Cash Balance: ${balance['cash_balance']:,.2f}")
        print(f"Position Value: ${balance['position_value']:,.2f}")
        print(f"Unrealized P&L: ${balance['unrealized_pnl']:,.2f}")
        print(f"Total Equity: ${balance['total_equity']:,.2f}")
        
        # Execute trade
        env.execute_trade(action, current_price, current_timestamp)
    
    # Close any open positions at the end
    env.close_all_positions(raw_data.iloc[-1]['close'], raw_data.iloc[-1]['timestamp'])
    
    # Calculate and display performance metrics
    metrics = env.get_performance_metrics()
    print("\nFinal Trading Performance Metrics:")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total Profit: ${metrics['total_profit']:,.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Plot equity curve
    equity_df = pd.DataFrame(env.equity_curve)
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df['timestamp'], equity_df['equity'], label='Total Equity')
    plt.plot(equity_df['timestamp'], equity_df['cash_balance'], label='Cash Balance')
    plt.plot(equity_df['timestamp'], equity_df['position_value'], label='Position Value')
    plt.title(f'Equity Curve - {env.stock_symbol}')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('equity_curve.png')

if __name__ == "__main__":
    run_mock_trading() 