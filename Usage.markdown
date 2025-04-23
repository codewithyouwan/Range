# Usage

To use this project, follow these steps:

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