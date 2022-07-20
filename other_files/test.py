import os
import numpy as np
import pandas as pd
from input_values import *

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, KFold, StratifiedKFold, cross_val_score, cross_validate, learning_curve, validation_curve
from lib_preprocessing import DO_PREPROCESSING


df, X, Y, X_train, X_test, y_train, y_test, X_train_part, X_valid, y_train_part, y_valid = DO_PREPROCESSING(drop_cols=False)
tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)

print(y_test.iloc[[1,2,3]])
i=0
for train_index, test_index in tscv.split(X):
  #print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  #print('test_index', test_index)
  aaa = Y.iloc[test_index]['y_true'].tolist()
  print(i, sum(aaa))
  i+=1
  #print(sum())
  #y_train, y_test = y[train_index], y[test_index]
a=98/0



def pena_media_mobile_prediction():
  # in LSTM
  window_size = 100
  N = len(v_all)

  run_avg_predictions = []
  run_avg_x = []

  mse_errors = []

  running_mean = 0.0
  run_avg_predictions.append(running_mean)

  decay = 0.5

  for pred_idx in range(1,N):

      running_mean = running_mean*decay + (1.0-decay)*v_all[pred_idx-1]
      run_avg_predictions.append(running_mean)
      print(pred_idx, '    ', running_mean, '    ', v_all[pred_idx-1])
      mse_errors.append((run_avg_predictions[-1]-v_all[pred_idx])**2)
      run_avg_x.append(date)

  print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))


  plt.figure(figsize = (18,9))
  plt.plot(range(len(v_all)),v_all,color='b',label='True')
  plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
  #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
  plt.xlabel('Date')
  plt.ylabel('Mid Price')
  plt.legend(fontsize=18)
  plt.show()
  

def get_file_path(N):
  file_name_base = 'RF'
  return 'results/' + str(ticker_to_predict) + '_' +str(PERIODS_TO_FORECAST) + '_' +file_name_base+'_' + str(N)+".xlsx"
print(get_file_path(1))
a=987/0

N = 1
PATH_ = 'results/'+str(PERIODS_TO_FORECAST) + '_' +file_name_base+'_' + str(N)+".xlsx"
print(PATH_)
while os.path.exists(PATH_) == True:
  print('exist true')
  N+=1
  PATH_ = 'results/'+str(ticker_to_predict) + '_' + str(PERIODS_TO_FORECAST) + '_' +file_name_base+'_' + str(N)+".xlsx"
print('N =>', N)
file_name = str(ticker_to_predict) + '_' + str(PERIODS_TO_FORECAST) + '_' + file_name_base + '_' + str(N)
print('file_name',file_name)

a=222/0

df, X_, Y_, X_train_, X_test_, y_train_, y_test_, X_train_part_, X_valid_, y_train_part_, y_valid_ = DO_PREPROCESSING()
CV = StratifiedKFold().split(X_, Y_)
for train_index, test_index in CV:
  print("TRAIN:", train_index, "TEST:", test_index)
  break
a=87/0


x = [int(x) for x in np.linspace(start = 20, stop = 220, num = 5)]
print(x)
a=987/0

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
df = pd.read_csv('SPY-1d.csv')
print(df)
X = df#['close'].tolist()


tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
for train_index, test_index in tscv.split(X):
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  #y_train, y_test = y[train_index], y[test_index]
a=98/0
'''TRAIN: [0] TEST: [1]
TRAIN: [0 1] TEST: [2]
TRAIN: [0 1 2] TEST: [3]
TRAIN: [0 1 2 3] TEST: [4]
TRAIN: [0 1 2 3 4] TEST: [5]'''
# Fix test_size to 2 with 12 samples
X = np.random.randn(12, 2)
y = np.random.randint(0, 2, 12)
tscv = TimeSeriesSplit(n_splits=3, test_size=2)
for train_index, test_index in tscv.split(X):
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
'''TRAIN: [0 1 2 3 4 5] TEST: [6 7]
TRAIN: [0 1 2 3 4 5 6 7] TEST: [8 9]
TRAIN: [0 1 2 3 4 5 6 7 8 9] TEST: [10 11]'''
# Add in a 2 period gap
tscv = TimeSeriesSplit(n_splits=3, test_size=2, gap=2)
for train_index, test_index in tscv.split(X):
  print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
'''TRAIN: [0 1 2 3] TEST: [6 7]
TRAIN: [0 1 2 3 4 5] TEST: [8 9]
TRAIN: [0 1 2 3 4 5 6 7] TEST: [10 11]'''