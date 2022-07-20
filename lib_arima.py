import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from datetime import date, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from input_values import *


def plot_ACF(v_hd, n_lags=None, label=''):
  from statsmodels.graphics import tsaplots
  #plot autocorrelation function
  #fig = tsaplots.plot_acf(v_hd, lags=50)
  n_lags = n_lags if type(n_lags) == int else 24
  fig = tsaplots.plot_acf(v_hd, lags=n_lags, color='g', title='ACF ' + str(label), zero=False) # title='Autocorrelation function '
  plt.show()

def plot_PACF(v_hd, n_lags=None, label=''):
  from statsmodels.graphics import tsaplots
  n_lags = n_lags if type(n_lags) == int else 24
  fig = tsaplots.plot_pacf(v_hd, lags=n_lags, color='g', title='PACF ' + str(label), zero=False) # title='Partial Autocorrelation function '
  plt.show()

def plot_ACF_PACF(v_hd, n_lags, series_name):
  plot_ACF(v_hd, n_lags=n_lags, label=' - ' + series_name)
  plot_PACF(v_hd, n_lags=n_lags, label=' - ' + series_name)



def find_best_model(v_hd, p_max, q_max):
  def get_v_pq_tuples(p_max, q_max):
    from itertools import product

    v_p = range(0, p_max+1, 1)
    v_q = range(0, q_max+1, 1)
    
    # Create a list with all possible combination of parameters
    parameters = product(v_p, v_q)
    parameters_list = list(parameters)
    parameters_list = [x for x in parameters_list if x != (0,0)]
    return parameters_list

  from cl_arima import c_Arima

  v_pq = get_v_pq_tuples(p_max, q_max)

  v_arima_results = []
  for i,pq in enumerate(v_pq):
    print('      ', i+1, '/', len(v_pq), '  |   ', pq)
    v_arima_results.append(c_Arima(v_hd, pq))

  R = v_arima_results
  df = pd.DataFrame({  'pq': [x.pq for x in R],   'aic': [x.aic for x in R],    'bic': [x.bic for x in R] })
  #print('AIC\n',df.sort_values(by=['aic', 'pq']))
  #print('BIC\n',df.sort_values(by=['bic', 'pq']))
  
  df['aic_o'], df['bic_o'] = df['aic'], df['bic']
  df = do_min_max_scaler(df, ['aic', 'bic'])
  df['aic_bic'] = (df['aic'] + df['bic']) / 2
  df = df.sort_values(by=['aic_bic', 'pq'], ascending=False)
  print('ARMA values, sorted by aic_bic')
  print(df)
  df.to_csv('results/'+str(ticker_to_predict) + '_ARMA_Train_best_model_'+str(p_max)+'-'+str(q_max)+'.csv')
  pq_best = df.iloc[0].pq
  return pq_best



def get_best_PQ_from_CSV(p_max, q_max):
  file_name = 'results/'+ str(ticker_to_predict) + '_ARMA_Train_best_model_'+ str(p_max) + '-' + str(q_max) + '.csv'
  df = pd.read_csv(file_name)
  best = df.iloc[0]['pq']
  s = best.split(', ')
  P, Q = int(s[0][-1]), int(s[1][0])
  PQ = (P,Q)
  return PQ



def do_min_max_scaler(df, v_cols):
  from sklearn.preprocessing import MinMaxScaler
  sc = MinMaxScaler(feature_range = (0, 1)) # TODO valuta se usare sc = StandardScaler()

  for col in v_cols:
    data_values = (df.filter([col])).values
    data_values = [abs(x) for x in data_values]
    v_scaled = sc.fit_transform(data_values)
    df[col] = v_scaled
  return df

