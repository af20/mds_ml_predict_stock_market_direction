'''
INIZIO
- Cerco il miglior modello (TRAIN), ottengo p,q <--- scelgo io   p_max, q_max
- Uso quel p,q in ==> do_kfold_cross_validation => produco 5 csv
- Uso quei 5 csv in ==> get_folds_accuracies ==> stampo per ogni fold (delle 5) l'Accuracy del test (uso tutto il dataset)
FINE

'''

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from datetime import date, datetime
import pandas as pd
import numpy as np

from lib_preprocessing import add_ticker_to_predict, DO_PREPROCESSING
from input_values import *
from lib_general_functions import *
from lib_arima import find_best_model, get_best_PQ_from_CSV

from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

DF_Origin, X_, Y_, X_train_, X_test_, y_train_, y_test_, X_train_part_, X_valid_, y_train_part_, y_valid_ = DO_PREPROCESSING(drop_cols=False)
print(DF_Origin)


X_SHAPE = X_.shape
Y_ = Y_[col_y_true].tolist()
y_train_ = y_train_[col_y_true].tolist()
y_test_ = y_test_[col_y_true].tolist()

T = lib_get_ind_or_ticker_from_id(ticker_to_predict, 'ticker')
series_name = T['name']
df_T = DF_Origin[series_name+' change']
df_T.index = pd.DatetimeIndex(df_T.index).to_period('M')

idx = int(TRAIN_SIZE*df_T.shape[0])
df_T_train = df_T.iloc[:idx]
df_T_test = df_T.iloc[idx:]

idx_valid = int(TRAIN_SIZE_on_VALIDATION*df_T_train.shape[0])
df_T_train_part = df_T_train[:idx_valid]
df_T_valid = df_T_train[idx_valid:]


v_hd = df_T.tolist()
v_hd_train = v_hd[:idx]
v_hd_test = v_hd[idx:]
v_hd_serie = DF_Origin[series_name].tolist()


class c_Arima:
  def __init__(self, v_hd, pq, v_hd_serie=None):
    # model_type sempre ARMA perchè differenzio sempre la serie storica
    self.v_hd = v_hd # contiene i ritorni
    self.v_hd_serie = v_hd_serie # contiene i prezzi

    model_type = 'ARMA'
    pdq=(pq[0], 0, pq[1])

    self.pq=pq
    self.pdq=pdq
    self.model_type=model_type
    
    self.model = ARIMA(v_hd, order=pdq) #  (0,0,1)   (p, d, q)
    self.results = self.model.fit(method_kwargs={"warn_convergence": False})

    #model.fit(start_params=[0, 0, 0, 1])

    self.residuals = pd.DataFrame(self.results.resid)
    self.aic = round(self.results.aic,2)
    self.bic = round(self.results.bic,2)
    #print('self.residuals')
    #print(self.residuals)
    #print('   pq ==> aic', self.aic, '    bic', self.bic)

    # REDDIT SPIEGAZ SU AIC   BIC  MIGLIORI  https://www.reddit.com/r/AskStatistics/comments/5ydt2c/if_my_aic_and_bic_are_negative_does_that_mean/

  def plot_ACF_PACF_series(self, n_lags=None):
    from lib_arima import plot_ACF_PACF
    T = lib_get_ind_or_ticker_from_id(ticker_to_predict, 'ticker')
    plot_ACF_PACF(self.v_hd, n_lags, T['name'])


  def plot_ACF_PACF_residuals(self, n_lags=None):
    from lib_arima import plot_ACF_PACF
    T = lib_get_ind_or_ticker_from_id(ticker_to_predict, 'ticker')
    plot_ACF_PACF(self.residuals, n_lags, T['name']+'residuals')


  def get_one_step_forecast(self):
    # one-step out-of sample forecast
    # array containing the forecast value, the standard error of the forecast, and the confidence interval information.
    FC = self.results.forecast()
    forecast = round(FC[0],6)
    return forecast
  


  def get_multi_step_forecast(self, pq, v_hd_train, v_hd_test):#start_index, end_index):

    v_hd_forecast = pd.Series(data=[], index=[])
    v_forecasts = []
    pdq = (pq[0], 0, pq[1])

    for i in range(len(v_hd_test)):
      model = ARIMA(v_hd_train, order=pdq) #  (0,0,1)   (p, d, q)   XXX
      results = model.fit(method_kwargs={"warn_convergence": False})
      forecast = results.forecast()#.predict(n_periods=1)#start=start_index, end=end_index)

      ser = pd.Series(data=[v_hd_test.iloc[i]], index=[v_hd_test.index[0]])
      #ser = pd.Series(data=[v_hd_test[i]], index=[v_hd_test[0]])
      v_hd_train = pd.concat([v_hd_train, ser])#forecast])
      
      forecast_value = round(forecast.values[0], 6)
      v_forecasts.append(forecast_value)
      #print('   ', i, '    forecast_value:', forecast_value)

      serF = pd.Series(data=[forecast_value], index=[v_hd_test.index[i]])
      v_hd_forecast = pd.concat([v_hd_forecast, serF])#forecast])


      print('   ',i, '   |   ', len(v_hd_test), '      ', forecast_value)
    pd.DataFrame(v_hd_forecast[1:]).to_csv('results/' + str(ticker_to_predict) + '_ARMA_forecast_'+str(pq)+'.csv')

    return v_hd_forecast
    # https://machinelearningmastery.com/make-sample-forecasts-arima-python/
    # https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7




  def do_kfold_cross_validation(self, pq, X):
    pdq = (pq[0], 0, pq[1])
    tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
    fold_no = 1
    acc_per_fold = []

    for train_index, test_index in tscv.split(X):
      v_hd_forecast = pd.Series(data=[], index=[])
      v_forecasts = []
      v_hd_train, v_hd_test = X.iloc[train_index], X.iloc[test_index]
      print('   fold_no:', fold_no, '   len(train_index):', len(train_index), '    len(test_index)', len(test_index))

      for i in range(len(v_hd_test)):
        model = ARIMA(v_hd_train, order=pdq) #  (0,0,1)   (p, d, q)   XXX
        results = model.fit(method_kwargs={"warn_convergence": False})
        forecast = results.forecast()#.predict(n_periods=1)#start=start_index, end=end_index)

        ser = pd.Series(data=[v_hd_test.iloc[i]], index=[v_hd_test.index[0]])
        #ser = pd.Series(data=[v_hd_test[i]], index=[v_hd_test[0]])
        v_hd_train = pd.concat([v_hd_train, ser])#forecast])
        
        forecast_value = round(forecast.values[0], 6)
        v_forecasts.append(forecast_value)
        #print('   ', i, '    forecast_value:', forecast_value)

        serF = pd.Series(data=[forecast_value], index=[v_hd_test.index[i]])
        v_hd_forecast = pd.concat([v_hd_forecast, serF])#forecast])

        #print('   ',i, '   |   ', len(v_hd_test), '      ', forecast_value)
      strpq = str(pq[0]) + '-' + str(pq[1])
      file_name = 'results/'+str(ticker_to_predict) + '_ARMA_' + strpq + '_fold_' + str(fold_no) + '.csv'
      pd.DataFrame(v_hd_forecast[1:]).to_csv(file_name)
      fold_no += 1



def get_folds_accuracies(pq):
  # su tutto il dataset, uso i csv prodotti con 'do_kfold_cross_validation'
  v_accuracies = []

  for fold_no in range(1, N_FOLDS+1):
    strpq = str(pq[0]) + '-' + str(pq[1])
    file_name = 'results/'+str(ticker_to_predict) + '_ARMA_' + strpq + '_fold_' + str(fold_no) + '.csv'
    
    df = pd.read_csv(file_name, names=['time', 'y_forecast'])
    df.dropna(inplace=True)
    df.set_index('time', inplace=True)
    df.index = pd.DatetimeIndex(df.index).to_period('M')

    DF = DF_Origin.copy()
    DF.index = pd.DatetimeIndex(DF.index).to_period('M')
    last_time_DF = DF.index[-1]
    df = df[df.index <= last_time_DF]
    DF = DF.loc[df.index]
    dt_from_pred, dt_from_to = DF.index[0], DF.index[-1]

    df = df.loc[DF.index]
    v_hd_forecast = df['y_forecast'].tolist()
    '''
      TODO 2 FUNZIONI
        1) Stampa la prediction nel grafico (passa da prediction a equity line)
        2) Aggiusta y_forecast => crea la sua variazione in N periodi (creo una cumulata) (se tra N periodi è > 0)
    '''
    y_true = DF['y_true'].tolist()
    
    y_pred = get_binary_prediction_with_treshold(v_hd_forecast, 0)
    print('  %pos:', np.mean(y_pred))
    accuracy_test = round(metrics.accuracy_score(y_true, y_pred),2)
    print('   fold_no:', fold_no, '     Pred:(' + str(dt_from_pred) + ' | ' + str(dt_from_to)+')    =>    Accuracy:', accuracy_test) # predico sempre 100 periodi
    fold_no +=1
    v_accuracies.append(accuracy_test)

'''
 - Find BEST MODEL .. Trovo il miglior pq    => pq = find_best_model()
      Creo: 'results/3539_ARMA_Train_best_model_2-2.csv'          df.to_csv('results/'+str(ticker_to_predict) + '_ARMA_Train_best_model_'+str(p_max)+'-'+str(q_max)+'.csv')
      => 'get_best_PQ_from_CSV' posso recuperare il risultato  
  - Creo la classe 'c_Arima' (pq)
  - CV FOLDS -> Folds Forecsst ... Faccio la cross-validation in N periodi    => do_kfold_cross_validation()   e creo i forecast delle Folds
        Creo: 'results/3539_ARMA_2-2_fold_N.csv'                  -> 'results/'+str(ticker_to_predict) + '_ARMA_' + strpq + '_fold_' + str(fold_no) + '.csv'

  - FOLD ACCURACIES ... Prendo il pq migliore, e Ottengo le accuracies in 5 folds  => get_folds_accuracies(pq, df_T, v_hd_test)
       leggo  'results/3539_ARMA_2-2_fold_N.csv'
       creo  'y_forecast' che paragonerò a y_true
       stampo i risultati nel terminale
'''

if __name__ == "__main__":
  pq=(0,1)
  get_folds_accuracies(pq)
  P=9876/0
  M = c_Arima(v_hd, pq, v_hd_serie)
  M.do_kfold_cross_validation(pq, df_T)
  a=22/0
  M.plot_ACF_PACF_series(60)
  M.plot_ACF_PACF_residuals(60)
  a=987/0
  p_max, q_max = 20,20
  pq = find_best_model(v_hd_train, p_max, q_max)
  pq = get_best_PQ_from_CSV(p_max, q_max)
  #pq = (2,2)
  
  a=987/0
  #print('pq best', pq)
  #v_hd_forecast = M.get_multi_step_forecast_mean(210, df_T_train, df_T_test)
  

  file_name = 'results/ARMA_'+str(pq)+'.csv'
  df = pd.read_csv(file_name, names=['time', 'y_forecast'])
  v_hd_forecast = df['y_forecast'].tolist()
  #v_hd_forecast = M.get_multi_step_forecast(pq, df_T_train, df_T_test)'''
  
  
  

  #M.get_multi_step_forecast(pq, v_hd_train, v_hd_test)


  '''
  a=987/0
  pq=(20,20)
  M = c_Arima(v_hd, pq)
  M._plot_ACF_residuals()
  M._plot_PACF_residuals()'''
