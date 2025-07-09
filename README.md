Stock Price Prediction Pipeline(Project)
=======================================

This project implements a complete pipeline for stock price prediction using an LSTM (Long Short-Term Memory) neural network. The pipeline fetches historical stock data, preprocesses it, trains an LSTM model, evaluates its performance, and visualizes the results. The project uses the `yfinance` library to fetch stock data (default: Apple, AAPL) and is implemented in Python.

Project Structure
----------------
- `stock_prediction_pipeline.py`: Main script containing the full pipeline.
- `lstm_model.h5`: Saved LSTM model (generated after running the pipeline).
- `scaler.pkl`: Saved MinMaxScaler object for data scaling (generated after running the pipeline).

Dependencies
------------
To run the pipeline, install the following Python libraries:
- yfinance
- numpy
- pandas
- scikit-learn
- tensorflow
- matplotlib

Install dependencies using pip:
```
pip install yfinance numpy pandas scikit-learn tensorflow matplotlib
```

Full Pipeline Description
------------------------
The pipeline consists of the following stages:

1. **Data Collection**
   - Fetches historical stock data (closing prices) for a specified ticker (default: AAPL) using the `yfinance` library.
   - Default date range: Last 1000 days (~3 years) up to the current date.
   - Output: Pandas DataFrame with closing prices.

2. **Data Preprocessing**
   - Scales the closing prices to the range [0, 1] using `MinMaxScaler`.
   - Creates sequences of 60 time steps (default) for LSTM input.
   - Splits data into 80% training and 20% testing sets.
   - Output: Training and testing datasets (X_train, X_test, y_train, y_test) and the scaler object.

3. **Model Building**
   - Constructs a Sequential LSTM model with:
     - Two LSTM layers (50 units each) with 20% dropout for regularization.
     - Two Dense layers (25 units and 1 unit) for output prediction.
   - Compiles the model with the Adam optimizer and Mean Squared Error (MSE) loss.
   - Output: Compiled LSTM model.

4. **Model Training**
   - Trains the model on the training data for 25 epochs (default) with a batch size of 32.
   - Uses 10% of the training data for validation during training.
   - Output: Trained model and training history.

5. **Model Evaluation**
   - Generates predictions on the test set.
   - Inverse transforms predictions and actual values to the original price scale.
   - Computes Root Mean Squared Error (RMSE) and R² score for evaluation.
   - Output: Predictions, RMSE, and R² score.

6. **Visualization**
   - Plots the training data, actual test prices, and predicted prices using Matplotlib.
   - Output: A line plot showing model performance over time.

7. **Model Saving**
   - Saves the trained LSTM model to `lstm_model.h5`.
   - Saves the scaler to `scaler.pkl` for future predictions.
   - Output: Saved model and scaler files.

Usage
-----
1. Save the script as `stock_prediction_pipeline.py`.
2. Ensure all dependencies are installed.
3. Run the script:
   ```
   python stock_prediction_pipeline.py
   ```
4. The script will:
   - Fetch AAPL stock data.
   - Train the LSTM model.
   - Display RMSE and R² metrics in the console.
   - Show a plot comparing training data, actual test prices, and predicted prices.
   - Save the model and scaler to disk.

Customization
------------
- **Ticker**: Change the `ticker` parameter in `run_pipeline(ticker='YOUR_TICKER')` (e.g., 'MSFT' for Microsoft).
- **Time Steps**: Adjust `time_steps` (default: 60) for different sequence lengths.
- **Epochs**: Modify `epochs` (default: 25) for training duration.
- **Date Range**: Update `start_date` and `end_date` in `run_pipeline()` for custom data ranges.

Example:
```python
run_pipeline(ticker='MSFT', time_steps=30, epochs=50)
```

Deployment Considerations
------------------------
To deploy this model in a production environment:
1. **API Integration**:
   - Wrap the model in a Flask or FastAPI endpoint to serve predictions.
   - Load the saved `lstm_model.h5` and `scaler.pkl` for inference.
2. **Data Pipeline**:
   - Automate data fetching with a scheduler (e.g., Apache Airflow).
   - Store historical data in a database (e.g., PostgreSQL).
3. **Monitoring**:
   - Log prediction errors and performance metrics.
   - Retrain the model periodically based on performance drift.
4. **Scalability**:
   - Use cloud services (e.g., AWS Lambda, GCP Cloud Functions) for serving.
   - Optimize inference with TensorFlow Serving or ONNX.

Notes
-----
- The pipeline uses LSTMs due to their effectiveness for time-series data. To experiment with Transformers, modify the model architecture using a library like `transformers`.
- Stock prices are non-stationary; consider adding features like trading volume or technical indicators (e.g., RSI, MA) for improved performance.
- The pipeline includes logging for debugging and monitoring.
- Ensure an active internet connection for `yfinance` to fetch data.

Future Improvements
-------------------
- Add hyperparameter tuning (e.g., using Keras Tuner).
- Incorporate additional features (e.g., volume, moving averages).
- Experiment with Transformer-based models for comparison.
- Implement cross-validation for more robust evaluation.

