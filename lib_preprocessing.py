import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from datetime import date, datetime
import pandas as pd
import numpy as np
from input_values import *
from lib_general_functions import *
import matplotlib.pyplot as plt
import seaborn as sns


def add_lags(df):
  if 'lags' not in [x['type'] for x in V_D_REGRESSORS]:
    return df

  T = lib_get_ind_or_ticker_from_id(ticker_to_predict, 'ticker')
  for lag in v_lags:
    #df['lag_'+str(lag)] = df[T['name_change']].shift(lag)
    df['lag_'+str(lag)] = df[T['name']].pct_change().shift(lag)

  return df



def add_price_ma(df, shift=None):
  shift = 1 if shift is None else shift  
  if 'price_ma' not in [x['type'] for x in V_D_REGRESSORS]:
    return df
  T = lib_get_ind_or_ticker_from_id(ticker_to_predict, 'ticker')

  for ma_period in d_price_ma['v_ma']:
    new_col = 'price_ma_' + str(ma_period)
    df[new_col] = df[T['name']].rolling(ma_period).mean()
    df[new_col] = df[T['name']] / df[new_col] - 1
    df[new_col] = df[new_col].shift(shift)
    df = Do_Transformations(d_price_ma['transformations'], df, new_col)
  return df


def add_roc(df, shift=None):
  shift = 1 if shift is None else shift  
  if 'roc' not in [x['type'] for x in V_D_REGRESSORS]:
    return df
  T = lib_get_ind_or_ticker_from_id(ticker_to_predict, 'ticker')
  for periods in d_roc['v_roc']:
    new_col = 'roc_' + str(periods)
    df[new_col] = df[T['name']].pct_change(periods=periods)
    df[new_col] = df[new_col].shift(shift)
    df = Do_Transformations(d_roc['transformations'], df, new_col)    
  return df



def add_tech_indicators(DF, shift=None):
  shift = 1 if shift is None else shift
  from finta import TA
  v_d_regressors_ind = [x['id'] for x in V_D_REGRESSORS if x['type'] == 'tech_indicator']
  T = lib_get_ind_or_ticker_from_id(ticker_to_predict, 'ticker')
  ind_df = get_ticker_df(ticker_to_predict)
  
  for x in v_d_tech_indicators:
    if x['id'] not in v_d_regressors_ind:
      continue
    periods = x['periods']
    df = eval('TA.' + x['indicator'] + '(ind_df, periods)')
    if not isinstance(df, pd.DataFrame):
      df = df.to_frame()
    df.rename(columns={df.columns[0]: x['name']}, inplace=True)
    df[x['name']] = df[x['name']].shift(shift)
    df = Do_Transformations(x['transformations'], df, x['name'])
    DF = DF.join(df)
  return DF


def add_macro_indicators(DF, shift=None):
  shift = 1 if shift is None else shift
  v_d_regressors_ind = [x['id'] for x in V_D_REGRESSORS if x['type'] == 'macro_indicator']
  for x in v_d_macro_indicators:
    if x['id'] not in v_d_regressors_ind:
      continue
    df = pd.read_csv(prefix_file_name_indicators+str(x['id'])+'.csv')[['time', 'close']]
    df = df.set_index('time')
    df.index = pd.DatetimeIndex(df.index)
    df['close'] = df['close'].shift(shift)
    df.rename(columns={"close": x['name']}, inplace=True)
    df = Do_Transformations(x['transformations'], df, x['name'])
    DF = DF.join(df)
  return DF


def get_ticker_df(id):
  T = lib_get_ind_or_ticker_from_id(id, 'ticker')
  df = pd.read_csv(prefix_file_name_tickers+str(id)+'.csv')
  df = df.set_index('time')
  df.drop(columns=['ticker'], inplace=True)
  df.index = pd.DatetimeIndex(df.index)#.strftime('%Y-%m-%d')
  return df



def apply_scaler(df, col, value_from=None, value_to=None):
  from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

  value_from = 0 if value_from is None else value_from
  value_to = 1 if value_to is None else value_to

  scR = RobustScaler()
  scSS = StandardScaler()
  scMM = MinMaxScaler(feature_range = (value_from, value_to))

  data_values = (df.filter([col])).values
  v_scaled = scR.fit_transform(data_values)
  v_scaled = scSS.fit_transform(v_scaled)
  v_scaled = scMM.fit_transform(v_scaled)
  df[col] = v_scaled
  return df


def get_ticker_data(ticker_to_predict):
  df = pd.read_csv(prefix_file_name_tickers+str(ticker_to_predict)+'.csv')[['time', 'close']]
  df = df.set_index('time')
  return df


def add_ticker_to_predict():
  T = lib_get_ind_or_ticker_from_id(ticker_to_predict, 'ticker')
  df = get_ticker_data(ticker_to_predict)
  df.rename(columns={'close': T['name']}, inplace=True)
  df.index = pd.DatetimeIndex(df.index)#.strftime('%Y-%m-%d')
  df[T['name_change']] = df[T['name']].pct_change()
  df[T['name_change']+'_pred'] = df[T['name']].pct_change(periods=PERIODS_TO_FORECAST).shift(-PERIODS_TO_FORECAST)
  return df


def Do_Transformations(v_transformations, df, col):

  for t in v_transformations:
    if 'ma_' in t: # OK
      periods = int(t.split('_')[1])
      #print('............MA............periods:', periods)
      df[col] = df[col].rolling(window=periods).mean()
    elif 'min_max' in t:
      S = t.split('__')[1].split('_')
      value_from, value_to = int(S[0]), int(S[1])
      #print('............min_max............col', col, '    value_from:', value_from, '  value_to', value_to)
      df = apply_scaler(df, col, value_from, value_to)
    elif 'roc_' in t: # FUNGE
      periods = int(t.split('_')[1])
      #print('............ROC............periods:', periods)
      df[col] = df[col].pct_change(periods=periods)
      #new_name = col + '(' + str(periods) + ')'
      #df.rename(columns={col: new_name}, inplace=True)


  return df
  

def add_special_indicators(df):
  v_d_regressors_ind = [x['id'] for x in V_D_REGRESSORS if x['type'] == 'special_indicator']
  for ind in v_d_special_indicators:
    if ind['id'] not in v_d_regressors_ind:
      continue
    name = ind['name']
    if ind['name'] == 'Mom+Sent':
      df[name] = df['roc_12'] * df['Sentiment']
    df = Do_Transformations(ind['transformations'], df, name)
  return df


def add_time_variables(df):
  df['month'] = pd.DatetimeIndex(df.index).month
  #df['quarter'] = pd.DatetimeIndex(df.index).quarter
  return df


def add_y_true(df, periods_forecast=None):
  periods_forecast = PERIODS_TO_FORECAST if periods_forecast is None else periods_forecast

  T = lib_get_ind_or_ticker_from_id(ticker_to_predict, 'ticker')
  #df[col_y_true] = np.where(df[T['name_change']+'_pred'] > 0, 1, 0)
  df[col_y_true] = np.where(df[T['name']].shift(-periods_forecast) - df[T['name']] > 0, 1, 0)
  return df


def do_over_sampling(X, y):
  from imblearn.over_sampling import SMOTE
  SM = SMOTE(random_state = 42)
  X, y = SM.fit_resample(X, y)
  return X, y


def get_train_valid_test_df(df):
  from sklearn.model_selection import train_test_split

  X = df.loc[:, df.columns != col_y_true]
  Y = df.loc[:, df.columns == col_y_true]
  X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=TRAIN_SIZE, shuffle=False)
  X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, train_size=TRAIN_SIZE_on_VALIDATION, shuffle=False)
  if DO_OVER_SAMPLING:
    X_train, y_train = do_over_sampling(X_train, y_train)
    X_train_part, y_train_part = do_over_sampling(X_train_part, y_train_part)
  return X, Y, X_train, X_test, y_train, y_test, X_train_part, X_valid, y_train_part, y_valid
  


def DO_PREPROCESSING(drop_cols=None, period_forecast=None):
  SHIFT = 1
  T = lib_get_ind_or_ticker_from_id(ticker_to_predict, 'ticker')
  df = add_ticker_to_predict()
  df = add_y_true(df, period_forecast)
  df = add_lags(df)
  df = add_macro_indicators(df, SHIFT)
  df = add_time_variables(df)
  df = add_roc(df, SHIFT)
  df = add_price_ma(df, SHIFT)
  df = add_tech_indicators(df, SHIFT)
  df = add_special_indicators(df)
  if drop_cols is not False:
    df.drop(columns=[T['name'], T['name_change'], T['name_change']+'_pred'], inplace=True)
  df_nan = df[df.isnull().any(1)]
  df.dropna(inplace=True)
  X, Y, X_train, X_test, y_train, y_test, X_train_part, X_valid, y_train_part, y_valid = get_train_valid_test_df(df)
  #df.drop(columns=['month', 'y_true', T['name_change'], T['name_change']+'_pred'], inplace=True)
  #df.to_excel('df_mine.xlsx')
  return df, X, Y, X_train, X_test, y_train, y_test, X_train_part, X_valid, y_train_part, y_valid

#df = DO_PREPROCESSING()
#df, X, Y, X_train, X_test, y_train, y_test, X_train_part, X_valid, y_train_part, y_valid = DO_PREPROCESSING(drop_cols=False)
#print(df)
#df = X
'''
# PRINT ALL COLUMNS ONE BY ONE, with S&P 500
T = lib_get_ind_or_ticker_from_id(ticker_to_predict, 'ticker')
for col in df.columns:
  print('col =>', col)
  if col == T['name']:
    continue
  lib_plot_2_lines(df.index.tolist(), {'name': T['name'], 'values': df[T['name']].tolist()}, {'name': col, 'values': df[col].tolist()})
  #print(df[col][10:30])
  #print(df[col])
  #plt.plot(df[col])
  #plt.title(col)
  #plt.show()
'''

'''
# DF DESCRIBE
print(df.describe())
'''

''' 
# PRINT EACH VARIABLE DISTRIBUTIONS (in GRID)
#df.hist(xlabelsize=0)
#plt.show()
'''

'''
#sns.pairplot(df); plt.show()
'''




'''
# CORRELAZIONE TRA
#  - Variazione dell'S&P da oggi ai successivi 12m 
#  - La variazione dell'ultimo periodo delle altra variabili

# Pre-Corr Matrix (Features Lag)
T = lib_get_ind_or_ticker_from_id(ticker_to_predict, 'ticker')
DF = df.copy()[[T['name_change']+ '_pred']]
v_cols = [x for x in df.columns if x not in [T['name'], T['name_change'], T['name_change'] + '_pred', 'y_true']]
df_shift = df.shift(PERIODS_TO_FORECAST)[v_cols]
df = pd.concat([DF, df_shift], axis=1)
v_cols_pct_change = ['ISM', 'CAPE', 'Sentiment', 'RSI 14', 'ATR 14 ROC']
for col in v_cols_pct_change:
  df[col] = df[col].pct_change()
df.dropna(inplace=True)
print(df)

# CORRELATION MATRIX 
fig, ax = plt.subplots(figsize=(20, 12)) # L, A
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax)
plt.show()
'''