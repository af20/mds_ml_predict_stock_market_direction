import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from datetime import date, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lib_preprocessing import add_ticker_to_predict, DO_PREPROCESSING
from input_values import *
from lib_general_functions import *
from lib_lstm import *
from lib_arima import do_min_max_scaler


def predict_test():
  predictions = M.predict(x_test)
  predictions = scaler.inverse_transform(predictions)
  rmse = np.sqrt(np.mean(predictions - y_test)**2)
  print('rmse', rmse)

  df_validation = pd.DataFrame({'Close': DF_Origin[symbol][len(v_train):]})
  df_validation['Predictions'] = predictions
  plt.figure(figsize=(16,8))
  plt.title('Model')
  plt.xlabel('Date')
  plt.ylabel('Close Price USD ($)')
  plt.plot(X_train_[symbol])
  plt.plot(df_validation[['Close', 'Predictions']])
  plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
  plt.show()

DF_Origin, X_, Y_, X_train_, X_test_, y_train_, y_test_, X_train_part_, X_valid_, y_train_part_, y_valid_ = DO_PREPROCESSING(drop_cols=False)
T = lib_get_ind_or_ticker_from_id(ticker_to_predict, 'ticker')
symbol = T['name']
v_all = DF_Origin[symbol].values
v_train = X_train_[symbol].values#.reshape(-1,1)
v_test = X_test_[symbol].values#reshape(-1,1)

v_all = do_exp_smooth(v_all)
v_train = do_exp_smooth(v_train)
v_test = do_exp_smooth(v_test)

v_all, scaler = scale_data(v_all)
v_train, scaler = scale_data(v_train)
v_test, scaler = scale_data(v_test)

x_train, y_train = get_train_xy(v_train)
x_test, y_test = get_test_xy(v_train, v_all)
#lib_plot_serie(y_test)

M = get_lstm_model(x_train, 1, print_summary=True)
M.compile(optimizer='adam', loss='mean_squared_error')
# optimizer = keras.optimizers.Adam(learning_rate=0.001) # model.compile(optimizer=optimizer, loss='mse')
M.fit(x_train, y_train, batch_size= 1, epochs=3)

# Predict beyond test data
# https://towardsdatascience.com/time-series-prediction-beyond-test-data-3f4625019fd9

from keras.preprocessing.sequence import TimeseriesGenerator
from pandas.tseries.offsets import DateOffset
n_input = 12
n_features = 1
generator = TimeseriesGenerator(X_train_, X_train_, length=n_input, batch_size=6)
print(generator)
history = M.fit_generator(generator, epochs=3, verbose=1)


pred_list = []
batch = X_train_[-n_input:].reshape((1, n_input, n_features))
for i in range(n_input):   
  pred_list.append(M.predict(batch)[0]) 
  batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)

print('pred_list', pred_list)
add_dates = [X_.index[-1] + DateOffset(months=x) for x in range(0,13) ]
future_dates = pd.DataFrame(index=add_dates[1:],columns=X_.columns)
print('future_dates', future_dates)

df_predict = pd.DataFrame(scaler.inverse_transform(pred_list), index=future_dates[-n_input:].index, columns=['Prediction'])
df_proj = pd.concat([X_, df_predict], axis=1)
print(df_proj)

import plotly.graph_objects as go
#import plotly.plotly as py

import plotly.offline as pyoff

plot_data = [
    go.Scatter(
        x=df_proj.index,
        y=df_proj['Sales'],
        name='actual'
    ),
    go.Scatter(
        x=df_proj.index,
        y=df_proj['Prediction'],
        name='prediction'
    )
]
plot_layout = go.Layout(
        title='Shampoo sales prediction'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

a=987/0

