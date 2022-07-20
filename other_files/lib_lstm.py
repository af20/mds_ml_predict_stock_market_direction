import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from datetime import date, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from input_values import *

def lib_plot_serie(v_x):
  plt.plot(v_x)
  plt.show()


def do_exp_smooth(v_x, plot_chart=None):
  # perform exponential moving average smoothing
  EMA = 0.0
  gamma = 0.75
  for ti in range(len(v_x)):
    EMA = gamma*v_x[ti] + (1-gamma)*EMA
    v_x[ti] = EMA
  if plot_chart == True:
    lib_plot_serie(v_x)
  return v_x




def scale_data(v_x, feature_range:tuple = None):
  from sklearn.preprocessing import MinMaxScaler
  feature_range = (0,1) if feature_range is None else feature_range
  scaler = MinMaxScaler(feature_range=feature_range)
  v_x_sc = scaler.fit_transform(v_x.reshape(-1,1))
  v_x_sc_resh = v_x_sc[0: len(v_x_sc), :]
  return v_x_sc_resh, scaler



def get_train_xy(v_train):
  x_train = []
  y_train = []
  for i in range(PERIODS_TO_FORECAST, len(v_train)):
      x_train.append(v_train[i-PERIODS_TO_FORECAST:i, 0])
      y_train.append(v_train[i, 0])
      
  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
  #print('x_train.shape', x_train.shape) # (851, 12, 1)
  return x_train, y_train


def get_test_xy(v_train, v_all):
  test_data = v_all[len(v_train)-PERIODS_TO_FORECAST: , : ]
  x_test = []
  y_test = v_all[len(v_train):]

  for i in range(PERIODS_TO_FORECAST, len(test_data)):
    x_test.append(test_data[i-PERIODS_TO_FORECAST:i, 0])

  x_test = np.array(x_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
  #print('x_test.shape', x_test.shape) # (216, 12, 1)
  return x_test, y_test


def get_lstm_model(x_train, N, print_summary=None):
  from tensorflow import keras
  from keras import layers
  M = keras.Sequential()
  if N == 1:
    M.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    M.add(layers.LSTM(100, return_sequences=False))
    M.add(layers.Dense(25)) 
    # Total params: 123,751
  if N == 2:
    pass
    #M.add(layers.LSTM(200, activation='relu', input_shape=(n_input, n_features)))
    #M.add(layers.Dropout(0.15))
  M.add(layers.Dense(1))
  if print_summary == True:
    M.summary()
  return M
