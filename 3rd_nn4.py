import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
data = pd.read_csv('BOH.csv')

# Parse dates
data['Date'] = pd.to_datetime(data['Date'])

# Normalize data
scaler = MinMaxScaler()
data[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close']])

# Define train-test split
split_index = int(len(data) * 0.8)
train = data[:split_index]
test = data[split_index:]


# Function to prepare data for LSTM
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)


# Number of time steps for LSTM
n_steps = 3

# Prepare data for LSTM
X_train, y_train = prepare_data(train[['Open', 'High', 'Low', 'Close']].values, n_steps)
X_test, y_test = prepare_data(test[['Open', 'High', 'Low', 'Close']].values, n_steps)

# Reshape data for LSTM input (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 4)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 4)

# Initialize LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 4)),
    Dense(4)
])

# Compile model
model.compile(optimizer='adam', loss='mse')

# Walk-forward validation
predictions = []
epoch_counter = 0

for i in range(len(X_test)):
    # Fit model
    model.fit(X_train, y_train, epochs=1, verbose=0)
    epoch_counter += 1

    # Print every 50 epochs
    if epoch_counter % 50 == 0:
        print(f"Trained epochs: {epoch_counter}")

    # Make prediction
    yhat = model.predict(X_test[i].reshape(1, n_steps, 4))

    # Store prediction
    predictions.append(yhat[0])

    # Update training data with the new observation
    new_observation = test[['Open', 'High', 'Low', 'Close']].values[i + n_steps]
    X_train = np.append(X_train, X_test[i].reshape(1, n_steps, 4), axis=0)
    y_train = np.append(y_train, [new_observation], axis=0)

# Convert predictions to array
predictions = np.array(predictions)

# Inverse transform to get real values
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("RMSE:", rmse)
print("MSE:", mse)
print("MAE:", mae)

# Plot predicted vs real values
plt.plot(test['Date'].iloc[n_steps:], predictions[:, 3], label='Predicted')
plt.plot(test['Date'].iloc[n_steps:], y_test[:, 3], label='Real')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('LSTM-DNN: Predicted vs Real')
plt.legend()
plt.show()
