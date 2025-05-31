import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load dataset
df = pd.read_csv("market_data.csv")

# Convert timestamp to datetime and sort in ascending order
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by="timestamp").reset_index(drop=True)

# Select features for model training
features = ["open", "high", "low", "close", "volume", "oi"]  # 6 features
target_col = "close"

# Normalize selected features
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Define sequence length
SEQ_LEN = 10

# Convert dataframe to sequences efficiently
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])  # Sequence of 10
        y.append(data[i + seq_length, 3])  # Predict next "close" price

    X, y = np.array(X), np.array(y)  # Convert lists to NumPy arrays
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Extract feature values as NumPy array
feature_values = df[features].values
X, y = create_sequences(feature_values, SEQ_LEN)

# Train-test split (80-20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {X_train.shape}, Testing samples: {X_test.shape}")

# Define Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, lstm_out):
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        attn_output = torch.bmm(attn_weights.transpose(1, 2), lstm_out)
        return attn_output[:, -1, :]  # Select last step

# Define BiLSTM Model with Attention
class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(BiLSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        output = self.fc(attn_out)
        return output

# Initialize model
input_dim = len(features)  # Set input_dim dynamically to match feature count
hidden_dim = 64
output_dim = 1
model = BiLSTMAttention(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
EPOCHS = 20
batch_size = 32
train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)

for epoch in range(EPOCHS):
    for batch in train_loader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "bilstm_attention_model.pth")
# Save the scaler
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved successfully!")
