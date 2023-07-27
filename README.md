# Range
Range prediction for EVs
This project is about analysing the data collected from currently active EVs and making a LSTM both 'single-variate' (taking only single variable(speed) as input to train and predict)
and 'multi-variate' (taking 2 variable input (speed and current) to train and predict) to make a more powerful model.

The mileage.ipynb file is the main file containing all the code.
The 'models' folders just contains the compiled and trained LSTM model both single and multi variate in .h5(HDF5 heirarchical data format) format.
Also the models are trained on a 3 month dataset, whereas tested on a 1month dataset.
