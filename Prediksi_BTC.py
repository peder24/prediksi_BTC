# Import library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam
from pandas.plotting import autocorrelation_plot, lag_plot
import datetime
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading
import yfinance as yf

# Download data Bitcoin dari Yahoo Finance (2014 hingga sekarang)
btc_data = yf.download('BTC-USD', start='2014-01-01')

print(f"Jumlah data: {len(btc_data)}")

# Menyimpan data ke CSV dengan mengatur ulang index
btc_data.reset_index(inplace=True)  # Reset index untuk menjadikan 'Date' sebagai kolom
btc_data.to_csv('BTC-USD.csv', index=False)
df = pd.read_csv("BTC-USD.csv")

# Display first few rows of the dataset
print("Preview of the dataset:")
print(df.head())

# 2. Exploratory Data Analysis - Deskripsi Variabel

# Get dataset dimensions
baris, kolom = df.shape
print(f'Jumlah baris: {baris}')
print(f'Jumlah kolom: {kolom}')

# Check data information (data types, non-null values)
print("\nInformasi Dataset:")
print(df.info())

# Statistical summary of the dataset
print("\nStatistik Deskriptif Dataset:")
print(df.describe())

# Check column names
print("\nKolom-kolom dalam dataset:")
print(df.columns.tolist())

# Explanation of each variable
print("\nPenjelasan Variabel:")
print("- Date: Tanggal observasi")
print("- Open: Harga pembukaan Bitcoin dalam USD")
print("- High: Harga tertinggi Bitcoin dalam USD pada hari tersebut")
print("- Low: Harga terendah Bitcoin dalam USD pada hari tersebut")
print("- Close: Harga penutupan Bitcoin dalam USD")
print("- Adj Close: Harga penutupan yang disesuaikan dalam USD")
print("- Volume: Jumlah Bitcoin yang diperdagangkan")

# 3. Exploratory Data Analysis - Menangani Missing Value dan Outliers

# Check for missing values
print("Jumlah missing value per kolom:")
print(df.isnull().sum())

# Check for duplicate data
print(f'Jumlah data duplikat: {df.duplicated().sum()}')

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Handle missing values if any
if df.isnull().sum().sum() > 0:
    print("Menangani missing value...")
    df_clean = df.dropna()
    print(f"Jumlah baris setelah menghapus missing value: {len(df_clean)}")
else:
    print("Tidak ada missing value dalam dataset.")
    df_clean = df.copy()

# Konversi semua kolom harga dan volume ke numerik, paksa error jadi NaN
for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

print("Cek tipe data kolom:")
print(df_clean.dtypes)
print("Jumlah missing value setelah konversi numerik:")
print(df_clean.isnull().sum())

# Drop semua baris yang masih ada NaN
df_clean = df_clean.dropna()    

# Set 'Date' as index
df_clean.set_index('Date', inplace=True)

# Check for outliers in 'Close' price using IQR method
Q1 = df_clean['Close'].quantile(0.25)
Q3 = df_clean['Close'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\nAnalisis Outlier pada Harga Penutupan (Close):")
print(f"Q1 (25%): {Q1}")
print(f"Q3 (75%): {Q3}")
print(f"IQR: {IQR}")
print(f"Batas bawah: {lower_bound}")
print(f"Batas atas: {upper_bound}")

outliers = df_clean[(df_clean['Close'] < lower_bound) | (df_clean['Close'] > upper_bound)]
print(f"Jumlah outlier: {len(outliers)}")
print(f"Persentase outlier: {len(outliers) / len(df_clean) * 100:.2f}%")

# Plot to visualize outliers
plt.figure(figsize=(10, 6))
plt.boxplot(df_clean['Close'])
plt.title('Box Plot of Bitcoin Closing Price')
plt.ylabel('Price (USD)')
plt.grid(True, alpha=0.3)
plt.show()

# 4. Exploratory Data Analysis - Univariate Analysis

# Univariate analysis of closing price
print("Univariate Analysis - Closing Price:")
print(df_clean['Close'].describe())

# Visualization of closing price over time
plt.figure(figsize=(14, 6))
plt.plot(df_clean.index, df_clean['Close'])
plt.title('Bitcoin Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True, alpha=0.3)
plt.show()

# Distribution of closing price
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_clean['Close'], bins=50, kde=True)
plt.title('Distribution of Bitcoin Closing Price')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.boxplot(y=df_clean['Close'])
plt.title('Box Plot of Bitcoin Closing Price')
plt.ylabel('Price (USD)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate skewness and kurtosis
skewness = skew(df_clean['Close'])
kurt = kurtosis(df_clean['Close'])
print(f"Skewness of Closing Price: {skewness:.4f}")
print(f"Kurtosis of Closing Price: {kurt:.4f}")
print(f"Interpretation: {'Positively skewed' if skewness > 0 else 'Negatively skewed'} distribution")
print(f"Interpretation: {'Heavy-tailed' if kurt > 0 else 'Light-tailed'} distribution compared to normal")

# Calculate daily returns
df_clean['Daily Return'] = df_clean['Close'].pct_change() * 100

# Visualize daily returns
plt.figure(figsize=(14, 6))
plt.plot(df_clean.index, df_clean['Daily Return'])
plt.title('Bitcoin Daily Returns (%)')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.grid(True, alpha=0.3)
plt.show()

# Distribution of daily returns
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df_clean['Daily Return'].dropna(), bins=50, kde=True)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return (%)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.boxplot(y=df_clean['Daily Return'].dropna())
plt.title('Box Plot of Daily Returns')
plt.ylabel('Daily Return (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate volatility (standard deviation of returns)
volatility = df_clean['Daily Return'].std()
print(f"Daily volatility (standard deviation of returns): {volatility:.4f}%")

# Autocorrelation analysis
plt.figure(figsize=(14, 6))
autocorrelation_plot(df_clean['Close'])
plt.title('Autocorrelation Plot of Bitcoin Closing Price')
plt.grid(True, alpha=0.3)
plt.show()

# Monthly analysis
monthly_data = df_clean.resample('M').mean()
plt.figure(figsize=(14, 6))
plt.plot(monthly_data.index, monthly_data['Close'], marker='o')
plt.title('Monthly Average Bitcoin Closing Price')
plt.xlabel('Date')
plt.ylabel('Average Price (USD)')
plt.grid(True, alpha=0.3)
plt.show()

# 5. Exploratory Data Analysis - Multivariate Analysis

# Calculate correlation matrix
correlation = df_clean[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
print("Correlation Matrix:")
print(correlation)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.4f')
plt.title('Correlation Matrix of Bitcoin Variables')
plt.show()

# Relationship between price and volume
plt.figure(figsize=(12, 6))
plt.scatter(df_clean['Volume'], df_clean['Close'], alpha=0.5, color='blue')
plt.title('Relationship Between Bitcoin Price and Trading Volume')
plt.xlabel('Volume')
plt.ylabel('Closing Price (USD)')
plt.grid(True, alpha=0.3)
plt.show()

# Calculate moving averages
df_clean['MA7'] = df_clean['Close'].rolling(window=7).mean()  # 7-day moving average
df_clean['MA30'] = df_clean['Close'].rolling(window=30).mean()  # 30-day moving average
df_clean['MA90'] = df_clean['Close'].rolling(window=90).mean()  # 90-day moving average

# Visualize closing price with moving averages
plt.figure(figsize=(14, 6))
plt.plot(df_clean.index, df_clean['Close'], label='Closing Price', alpha=0.7)
plt.plot(df_clean.index, df_clean['MA7'], label='7-day MA', color='red')
plt.plot(df_clean.index, df_clean['MA30'], label='30-day MA', color='green')
plt.plot(df_clean.index, df_clean['MA90'], label='90-day MA', color='orange')
plt.title('Bitcoin Closing Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 6. Data Preparation

# Normalize data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_clean[['Close']])

# Define time steps (lag)
time_step = 60  # Using 60 days of data to predict the next day
print(f"Time step (lag): {time_step} days")

# Split data into training and testing sets
train_size = int(len(df_clean) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

print(f"Training data size: {train_size} days")
print(f"Testing data size: {len(df_clean) - train_size} days")

# Function to create dataset with time steps
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Create datasets with time_step for training and testing
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape data for LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Display data dimensions
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# 7. Modeling - LSTM Model

# Build LSTM model
lstm_model = Sequential(name='lstm_bitcoin_forecast')

lstm_model.add(LSTM(150, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dropout(0.3))

lstm_model.add(LSTM(100, return_sequences=False))
lstm_model.add(Dropout(0.3))

lstm_model.add(Dense(50))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
lstm_model.summary()

# Train the model
lstm_history = lstm_model.fit(X_train,
                             y_train,
                             epochs=25,
                             batch_size=32,
                             validation_data=(X_test, y_test),
                             verbose=1)

# Plot loss values
plt.figure(figsize=(14, 6))
plt.plot(lstm_history.history['loss'], label='Training Loss')
plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Model: Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 8. Evaluation

# Make predictions
lstm_predictions = lstm_model.predict(X_test)

# Inverse transform predictions and actual values to original scale
lstm_predictions = scaler.inverse_transform(lstm_predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate evaluation metrics for LSTM
lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_predictions))
lstm_mae = mean_absolute_error(y_test_actual, lstm_predictions)
lstm_r2 = r2_score(y_test_actual, lstm_predictions)
lstm_mape = np.mean(np.abs((y_test_actual - lstm_predictions) / y_test_actual)) * 100

# Print evaluation results
print("\nLSTM Model Evaluation Results:")
print(f"RMSE: {lstm_rmse:.2f} USD")
print(f"MAE: {lstm_mae:.2f} USD")
print(f"R²: {lstm_r2:.4f}")
print(f"MAPE: {lstm_mape:.2f}%")

# Visualize predictions vs actual values
test_dates = df_clean.index[train_size + time_step + 1:train_size + time_step + 1 + len(y_test_actual)]

plt.figure(figsize=(16, 8))
plt.plot(test_dates, y_test_actual, label='Actual Price', color='blue')
plt.plot(test_dates, lstm_predictions, label='LSTM Prediction', color='red', alpha=0.7)
plt.title('Bitcoin Price Prediction with LSTM Model')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Zoom in on a portion of the test data for better visualization
zoom_start = int(len(test_dates) * 0.7)
zoom_end = int(len(test_dates) * 0.9)

plt.figure(figsize=(16, 8))
plt.plot(test_dates[zoom_start:zoom_end], y_test_actual[zoom_start:zoom_end], label='Actual Price', color='blue')
plt.plot(test_dates[zoom_start:zoom_end], lstm_predictions[zoom_start:zoom_end], label='LSTM Prediction', color='red', alpha=0.7)
plt.title('Zoomed Comparison of LSTM Predictions vs Actual Bitcoin Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate prediction error
prediction_error = y_test_actual - lstm_predictions

# Plot prediction error
plt.figure(figsize=(16, 6))
plt.plot(test_dates, prediction_error, color='green')
plt.axhline(y=0, color='r', linestyle='-')
plt.title('LSTM Model Prediction Error')
plt.xlabel('Date')
plt.ylabel('Error (Actual - Predicted) in USD')
plt.grid(True, alpha=0.3)
plt.show()

# Plot error distribution
plt.figure(figsize=(12, 6))
sns.histplot(prediction_error, bins=50, kde=True)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error (USD)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Calculate error statistics
error_mean = np.mean(prediction_error)
error_std = np.std(prediction_error)
print(f"\nError Statistics:")
print(f"Mean Error: {error_mean:.2f} USD")
print(f"Standard Deviation of Error: {error_std:.2f} USD")

# 9. Future Price Prediction

# Get the last 60 days of data
x_input = scaled_data[-time_step:].reshape(1, time_step, 1)

# List to store predictions
temp_input = list(x_input[0, :, 0])
output_list = []

# Predict for the next 30 days
n_steps = 30
for i in range(n_steps):
    if len(temp_input) > time_step:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, time_step, 1)
        yhat = lstm_model.predict(x_input, verbose=0)
        temp_input.append(yhat[0, 0])
        temp_input = temp_input[1:]
        output_list.append(yhat[0, 0])
    else:
        x_input = np.array(temp_input).reshape(1, time_step, 1)
        yhat = lstm_model.predict(x_input, verbose=0)
        temp_input.append(yhat[0, 0])
        output_list.append(yhat[0, 0])

# Convert predictions to original scale
future_predictions = scaler.inverse_transform(np.array(output_list).reshape(-1, 1))

# Create dates for future predictions
last_date = df_clean.index[-1]
prediction_dates = pd.date_range(start=last_date + datetime.timedelta(days=1), periods=n_steps)

# Visualize future predictions
plt.figure(figsize=(16, 8))
plt.plot(df_clean.index[-100:], df_clean['Close'].values[-100:], label='Historical Price', color='blue')
plt.plot(prediction_dates, future_predictions, label='Predicted Price', color='red', linestyle='--', marker='o')
plt.title(f'Bitcoin Price Prediction for Next {n_steps} Days')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Display future price predictions
future_df = pd.DataFrame({
    'Date': prediction_dates,
    'Predicted_Price': future_predictions.flatten()
})
future_df.set_index('Date', inplace=True)
print("\nPredicted Bitcoin Prices for the Next 30 Days:")
print(future_df)

# Calculate expected price change
initial_price = df_clean['Close'].iloc[-1]
final_predicted_price = future_predictions[-1][0]
price_change = final_predicted_price - initial_price
price_change_percent = (price_change / initial_price) * 100

print(f"\nCurrent Bitcoin Price (Last Date in Dataset): ${initial_price:.2f}")
print(f"Predicted Bitcoin Price after 30 Days: ${final_predicted_price:.2f}")
print(f"Expected Price Change: ${price_change:.2f} ({price_change_percent:.2f}%)")