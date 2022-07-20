import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# https://towardsdatascience.com/time-series-prediction-beyond-test-data-3f4625019fd9

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


import random
import tensorflow as tf
from tensorflow import keras
from keras import layers
from numpy.random import seed

seed_value = 20 # 0: su     1: giu     3: giu poco
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


DF_Origin, X_, Y_, X_train_, X_test_, y_train_, y_test_, X_train_part_, X_valid_, y_train_part_, y_valid_ = DO_PREPROCESSING(drop_cols=False)
T = lib_get_ind_or_ticker_from_id(ticker_to_predict, 'ticker')
symbol = T['name']

def parser(x):
    return pd.datetime.strptime('190'+x, '%Y-%m')
df = pd.read_csv('data/shampoo.csv', parse_dates=[0], index_col=0, date_parser=parser)
#print(DF_Origin)
T = lib_get_ind_or_ticker_from_id(ticker_to_predict, 'ticker')
symbol = T['name']#_change']

df = train = pd.DataFrame({symbol: DF_Origin[symbol]})#df
df = train = df[:500]

from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from pandas.tseries.offsets import DateOffset

scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)
n_input = 12
n_features = 1
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)


model = keras.Sequential()
model.add(layers.LSTM(200, activation='relu', input_shape=(n_input, n_features)))
model.add(layers.Dropout(0.15))
model.add(layers.Dense(1))
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')
history = model.fit_generator(generator, epochs=3, verbose=1)

# TODO rolling predictions and v_y_pred .... usa time series split, rifitta
pred_list = []
batch = train[-n_input:].reshape((1, n_input, n_features))
for i in range(n_input):   
  pred_list.append(model.predict(batch)[0]) 
  batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)


add_dates = [df.index[-1] + DateOffset(months=x) for x in range(0,n_input+1) ]
future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)
print('future_dates\n', future_dates)


df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=future_dates[-n_input:].index, columns=['Prediction'])
df_proj = pd.concat([df,df_predict], axis=1)

print('df_predict\n', df_predict)
print('df_proj\n', df_proj)


plt.plot(df_proj.index, df_proj[symbol], label='actual')
plt.plot(df_proj.index, df_proj['Prediction'], label='prediction')
plt.title(symbol+' prediction')
plt.show()

