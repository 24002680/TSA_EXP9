# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      from statsmodels.tsa.arima.model import ARIMA
      from sklearn.metrics import mean_squared_error
      
      # Load dataset
      data = pd.read_csv(r"C:\Users\admin\Downloads\favorite_music_dataset.csv")
      
      # Convert date column to datetime and set index
      data['Listened_Date'] = pd.to_datetime(data['Listened_Date'])
      data.set_index('Listened_Date', inplace=True)
      
      # Aggregate by date - count of songs listened per day
      daily_counts = data.resample('D').size().rename('Count')
      
      def arima_model(data, order):
          train_size = int(len(data) * 0.8)
          train_data, test_data = data[:train_size], data[train_size:]
      
          model = ARIMA(train_data, order=order)
          fitted_model = model.fit()
      
          forecast = fitted_model.forecast(steps=len(test_data))
      
          rmse = np.sqrt(mean_squared_error(test_data, forecast))
      
          plt.figure(figsize=(10, 6))
          plt.plot(train_data.index, train_data, label='Training Data')
          plt.plot(test_data.index, test_data, label='Testing Data')
          plt.plot(test_data.index, forecast, label='Forecasted Data')
          plt.xlabel('Date')
          plt.ylabel('Count of Songs Listened')
          plt.title('ARIMA Forecasting for Daily Song Count')
          plt.legend()
          plt.show()
      
          print("Root Mean Squared Error (RMSE):", rmse)
          arima_model(daily_counts, order=(5,1,0))
### OUTPUT:



<img width="1215" height="720" alt="Screenshot 2025-10-28 211828" src="https://github.com/user-attachments/assets/a0378e4b-e788-4214-b742-66d175265004" />




### RESULT:
Thus the program run successfully based on the ARIMA model using python.
