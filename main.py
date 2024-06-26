import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Function to preprocess input data
def preprocess_data(df):
    # Drop unnecessary columns and normalize data
    features = ['Pressure [mbar]', '98m WV [°]', '78m WV [°]', '48m WV [°]', 'Temp 5m [°C]', 'Hum 5m [%]']
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[features])
    return X, scaler

# Function to create sequences from input data
def create_sequences(X, time_steps=1):
    Xs = []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i:i + time_steps, :])
    return np.array(Xs)

# Function to load and predict using the LSTM model
def predict_wind_speed(input_file, output_file):
    # Load input data
    df_input = pd.read_excel(input_file)

    # Preprocess data
    X, scaler = preprocess_data(df_input)

    # Load the LSTM model
    model = load_model('lstm_model_improved.h5')

    # Prepare sequences
    time_steps = 10  # number of past time steps to use for prediction
    X_seq = create_sequences(X, time_steps)

    # Predictions
    y_pred = model.predict(X_seq)

    # Inverse transform predictions
    y_pred_inv = scaler.inverse_transform(y_pred)

    # Add predictions to the input dataframe
    df_input['Predicted Wind Avg Speed'] = y_pred_inv

    # Save output to Excel file
    df_input.to_excel(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Example usage
if __name__ == '__main__':
    input_file = 'input_data.xlsx'  # Update with your input file path
    output_file = 'predicted_wind_speed.xlsx'  # Update with desired output file path
    predict_wind_speed(input_file, output_file)
