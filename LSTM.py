
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import pickle
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#  Data Collection
def fetch_stock_data(ticker, start_date, end_date):

   # Fetch historical stock data using yfinance.
    try:
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError("No data retrieved. Check ticker or date range.")
        return stock_data[['Close']]
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise

# Data Preprocessing
def preprocess_data(data, time_steps=60, train_split=0.8):
    logger.info("Preprocessing data")
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))  ## Convert the data in range between (0,1)
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split data in train and test data
    train_size = int(len(X) * train_split)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, scaler

# Model Building
def build_lstm_model(time_steps, units=50, dropout_rate=0.2): 
     # Build and compile an LSTM model.
     # input_shape = (time_steps, 1)
    logger.info("Building LSTM model")
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(time_steps, 1)))  # give a output 50 dim vector 
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Model Training
def train_model(model, X_train, y_train, epochs=25, batch_size=32, validation_split=0.1):
    """
    Train the LSTM model.
    """
    logger.info("Training model")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    return model, history

# Model Evaluation
def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate the model using RMSE and R2 score.
    Returns metrics and predictions.
    """
    logger.info("Evaluating model")
    predictions = model.predict(X_test)
    
    # Inverse transform for original scale
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
    r2 = r2_score(y_test_scaled, predictions)
    
    logger.info(f"RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")
    return predictions, rmse, r2

# 6. Visualization
def plot_results(train_data, test_data, predictions, scaler, ticker):
    """
    Plot training, actual, and predicted prices.
    """
    logger.info("Plotting results")
    train_data = scaler.inverse_transform(train_data.reshape(-1, 1))
    test_data = scaler.inverse_transform(test_data.reshape(-1, 1))
    
    plt.figure(figsize=(14, 5))
    plt.plot(train_data, label='Training Data')
    plt.plot(range(len(train_data), len(train_data) + len(test_data)), test_data, label='Actual Price')
    plt.plot(range(len(train_data), len(train_data) + len(predictions)), predictions, label='Predicted Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

#  Save Model and Scaler
def save_model_and_scaler(model, scaler, model_path='lstm_model.h5', scaler_path='scaler.pkl'):

    logger.info("Saving model and scaler")
    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

# Main Pipeline
def run_pipeline(ticker='AAPL', start_date=None, end_date=None, time_steps=60, epochs=25):
    """
    Execute the full pipeline.
    """
    logger.info("Starting pipeline")
    
    # Set default dates
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=1000)
    
    try:
        # Data Collection
        data = fetch_stock_data(ticker, start_date, end_date)
        
        # Preprocessing
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data.values, time_steps)
        
        # Model Building
        model = build_lstm_model(time_steps)
        
        # Model Training
        model, history = train_model(model, X_train, y_train, epochs)
        
        # Model Evaluation
        predictions, rmse, r2 = evaluate_model(model, X_test, y_test, scaler)
        
        # Visualization
        plot_results(y_train, y_test, predictions, scaler, ticker)
        
        # Save Model
        save_model_and_scaler(model, scaler)
        
        logger.info("Pipeline completed successfully")
        return model, scaler, history, rmse, r2
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

# Run the pipeline
if __name__ == "__main__":
    run_pipeline(ticker='AAPL', time_steps=60, epochs=25)
