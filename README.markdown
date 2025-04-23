# Vehicle Mileage Prediction using LSTM

## Introduction

This project aims to predict vehicle mileage using deep learning techniques, specifically an LSTM (Long Short-Term Memory) model, based on historical data. The data includes information such as date, time, speed, battery voltage, battery current, cumulative discharge, latitude, longitude, and direction. By leveraging LSTM, the model captures temporal dependencies in the data to make accurate mileage predictions.

## Project Structure

The project is structured into several key components:

1. **Data Preprocessing**: Functions to clean and transform the raw data.
2. **Data Class**: Manages the main dataframe, performs sorting, calculates energy consumption, and filters invalid data.
3. **LSTM Class**: Prepares windowed data, splits it into training and testing sets, and trains or loads the LSTM model.
4. **Model Training and Evaluation**: Trains the LSTM model and optionally visualizes the results.

## Data Preprocessing

The following functions are used for data preprocessing:

- `to_seconds(str)`: Converts a time string (e.g., "HH:MM:SS") to total seconds.
- `string_split(str)`: Converts a date string (e.g., "DD/MM/YY") to a datetime object.
- `month(str)`: Extracts the month name from a date string.
- `indexDates(date)`: Extracts the day from a date string.
- `calDistanceLatLong(lat1, lat2, delLong)`: Calculates the distance between two geographical points using their latitudes and longitudes.

Utility functions include:

- `mean_squared_error(true, pred)`: Calculates the mean squared error between actual and predicted values.
- `findBestFitFunction(x, y, largestDegree)`: Finds the best polynomial fit for given data up to a specified degree.

## Data Class

The `Data` class is initialized with a dataframe and performs the following operations:

- Sorts data by date and time.
- Calculates distances between consecutive geographical points.
- Computes energy consumption.
- Filters out invalid data (e.g., zero latitude/longitude, negative energy consumption, distances &gt; 170 meters, or zero/negative battery current).
- Calculates time differences, average speeds, and mileage.

A secondary dataframe (`df2`) is created for model training, containing features like date, time, mileage, speed, and battery current.

## LSTM Class

The `LSTM` class prepares data for the LSTM model and manages training or loading. Key features include:

- **Windowed Data**: Converts the dataframe into a windowed format suitable for LSTM input.
- **Data Splitting**: Splits windowed data into training and testing sets.
- **Model Architecture**: Defines an LSTM model with layers (e.g., LSTM with 64 units, Dense layers with ReLU activation) and compiles it with mean squared error loss and the Adam optimizer.
- **Training**: Trains the model and optionally plots actual vs. predicted values.

The class supports training a new model or loading a pre-trained one for further use.

## Model Training and Evaluation

The `train` method in the `LSTM` class:

- Splits data into training and validation sets.
- Trains the model for a specified number of epochs (e.g., 100).
- Optionally plots actual and predicted mileage against speed, including a trend line from `findBestFitFunction`.

The project also demonstrates loading a pre-trained model and making predictions, as shown in the final code cell.

## Installation

To set up the project, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment** (Optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Jupyter Notebook** (If not already installed):

   ```bash
   pip install notebook
   ```

## Usage

To use this project:

1. **Data Preparation**: Ensure your data is in a CSV file with columns: `date`, `time`, `speed`, `batVolt`, `batCurrent`, `cummulativeDischarge`, `lat`, `long`, `longDirection`, `latDirection`.

2. **Initialize Data Class**:

   ```python
   import pandas as pd
   from data import Data  # Assuming the Data class is in data.py
   
   columns_to_pick = ['date', 'time', 'speed', 'batVolt', 'batCurrent', 'cummulativeDischarge', 'lat', 'long', 'longDirection', 'latDirection']
   df = pd.read_csv('1monthdata.csv', usecols=columns_to_pick)
   OneMonth = Data(df)
   ```

3. **Train or Load Model**:

   ```python
   from lstm import LSTM  # Assuming the LSTM class is in lstm.py
   
   # Train a new model
   trained_model_multiVariate = LSTM(dataframe=OneMonth.df2, fileName='model.h5', train=True, partiallyTrained=False, size=3, percentage_train=0.8, plot=True, variate='Multi')
   
   # Or load a pre-trained model
   # trained_model_multiVariate = LSTM(dataframe=OneMonth.df2, fileName='model.h5', train=False, partiallyTrained=True, size=3, percentage_train=0.8, plot=True, variate='Multi')
   ```

4. **Make Predictions**:

   ```python
   windowed = LSTM.toWindowedDf(df=OneMonth.df2, size=trained_model_multiVariate.sizeOfWindow, variate='Multi')
   x_test, y_test = LSTM.splitData(percentage_train=-1, size=trained_model_multiVariate.sizeOfWindow, windowed_df=windowed, training=False, variate='Multi')
   predicted = trained_model_multiVariate.model.predict(x_test).flatten()
   ```

## Dependencies

The project requires the following Python libraries:

- `numpy==1.21.0`
- `pandas==1.3.0`
- `matplotlib==3.4.2`
- `tensorflow==2.5.0`

These dependencies are listed in the `requirements.txt` file and can be installed using `pip install -r requirements.txt`.

## Conclusion

This project provides a framework for predicting vehicle mileage using LSTM models. By following the steps outlined above, users can preprocess their data, train an LSTM model, and generate predictions for new data points.