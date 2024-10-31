import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
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

# Function to prepare data for regression
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps].flatten())
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Number of time steps for regression
n_steps = 3

# Prepare initial training data for regression
X_train, y_train = prepare_data(train[['Open', 'High', 'Low', 'Close']].values, n_steps)
X_test, y_test = prepare_data(test[['Open', 'High', 'Low', 'Close']].values, n_steps)

# Initialize Linear Regression model
model = LinearRegression()

# Walk-forward validation
predictions = []

for i in range(len(X_test)):
    # Fit model on the training data
    model.fit(X_train, y_train)

    # Make prediction for the current test instance
    yhat = model.predict(X_test[i].reshape(1, -1))

    # Store prediction
    predictions.append(yhat[0])

    # Update training data with the new observation
    if i + n_steps < len(test):
        new_observation = test[['Open', 'High', 'Low', 'Close']].values[i + n_steps]
        X_train = np.append(X_train, X_test[i].reshape(1, -1), axis=0)
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
plt.title('Linear Regression: Predicted vs Real')
plt.legend()
plt.show()
