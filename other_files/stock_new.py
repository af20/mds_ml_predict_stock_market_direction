import os
import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta as rd
from library import *

import yfinance as yf
from finta import TA

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics




'''
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU
from keras import Input

'''
import xgboost as xgb


class Ticker():

  INTERVAL = '1d'     # Sample rate of historical data
  NUM_PERIODS = 3000000     # The number of periods of historical data to retrieve

  # List of symbols for technical indicators
  INDICATORS = ['RSI', 'MACD', 'STOCH', 'ATR', 'MOM', 'MFI', 'ROC', 'CCI', 'EMV', 'VORTEX']
  IND_MA_periods = 7
  COL_STATS = CS = 'Adj Close'
  TRAIN_SIZE = 0.8
  TRAIN_SIZE_on_VALIDATION = 0.75
  #v_remove = ['MACD']
  


  def __init__(self, symbol):

      """
      Constructor for class
      Will obtain historical data for NUM_PERIODS periods (days, weeks, months)
      :param symbol: ticker of stock
      """

      self.symbol = symbol
      self._get_historical_data()

  def _get_historical_data(self):

      """
      Function that uses the yfinance API to get stock data
      :return:
      """

      #start = (datetime.date.today() - datetime.timedelta( self.NUM_PERIODS) )
      #end = datetime.datetime.today()
      #self.data = yf.download(self.symbol, end=end, interval=self.INTERVAL)
      #self.data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)
      #self.data.to_csv(self.symbol + '-' + self.INTERVAL+'.csv')#, index=False)
      self.data = pd.read_csv(self.symbol + '-' + self.INTERVAL+'.csv')
      MINr = max(0, self.data.shape[0]-self.NUM_PERIODS)
      MAXr = self.data.shape[0]
      self.data = self.data.iloc[MINr:MAXr]


  def _exponential_smooth(self, alpha):

      """
      Function that exponentially smooths dataset so values are less 'rigid'
      :param alpha: weight factor to weight recent values more
      0 < alpha <= 1
      """
      self.data = self.data.ewm(alpha=alpha).mean()
      #self.data['Adj Close'].plot(); plt.show()


  def _get_indicator_data(self):

      """
      Function that uses the finta API to calculate technical indicators used as the features
      :return:
      """
      import matplotlib.pyplot as plt


      for indicator in self.INDICATORS:
        ind_data = eval('TA.' + indicator + '(self.data)')
        if not isinstance(ind_data, pd.DataFrame):
          ind_data = ind_data.to_frame()
        ind_data = ind_data.rolling(window=self.IND_MA_periods).mean()
        #print('indicator:', indicator) print(ind_data) print()
        #ind_data.plot() plt.show()
        self.data = self.data.merge(ind_data, left_index=True, right_index=True)
      self.data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)
      self.data.rename(columns={"SIGNAL": 'MACD Signal'}, inplace=True)
      self.data.rename(columns={'VIm': 'Vortex VIm'}, inplace=True)
      
      del (self.data['MACD'])
      del (self.data['VIp'])

      # Also calculate moving averages for features
      self.data['ema50'] = self.data[self.CS] / self.data[self.CS].ewm(50).mean()
      self.data['ema21'] = self.data[self.CS] / self.data[self.CS].ewm(21).mean()
      self.data['ema14'] = self.data[self.CS] / self.data[self.CS].ewm(14).mean()
      self.data['ema5'] = self.data[self.CS] / self.data[self.CS].ewm(5).mean()


  def _produce_prediction(self, prediction_window=10):

      """
      Function that produces the 'truth' values
      At a given row, it looks 'window' rows ahead to see if the price increased (1) or decreased (0)
      :param window: number of days, or rows to look ahead to see what the price did
      """
      window = prediction_window
      prediction = (self.data.shift(-window)[self.CS] >= self.data[self.CS])
      prediction = prediction.iloc[:-window]
      self.data['pred'] = prediction.astype(int)
  

  def _min_max_scaler(self):
    sc = MinMaxScaler(feature_range = (0, 1)) # TODO valuta se usare sc = StandardScaler()

    for col in self.data.columns:
      data_values = (self.data.filter([col])).values
      v_scaled = sc.fit_transform(data_values)
      self.data[col] = v_scaled


  def _produce_data(self, prediction_window):

      """
      Main data function that calls the others to smooth, get features, and create the predictions
      :param window: value used to determine the prediction
      :return:
      """
      self._exponential_smooth(0.5)#65) # 0.9)
      self._get_indicator_data() # <--- qui inserisce tutti gli indicatori
      self._produce_prediction(prediction_window=prediction_window)
      self.data_origin = self.data.copy()

      # Remove columns that won't be used as features
      del (self.data['open'])
      del (self.data['high'])
      del (self.data['low'])
      del (self.data['volume'])
      del (self.data['close'])
      del (self.data['Adj Close'])
      self.data = self.data.dropna()

      self._min_max_scaler()


  def _split_data(self):
      """Function to partition the data into the train and test set"""
      self.y = self.data['pred']
      features = [x for x in self.data.columns if x not in ['pred']]
      self.X = self.data[features]

      self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(self.X, self.y, train_size=self.TRAIN_SIZE, random_state=42) # train_size=int(self.TRAIN_SIZE*len(self.X))    |    2 * len(self.X) // 3)
      self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train_full, self.y_train_full, train_size=self.TRAIN_SIZE_on_VALIDATION, random_state=42)



  def _get_baseline_one(self):
    one_predicion_train = [1 for x in range(self.X_train_full.shape[0])]
    one_predicion_test = [1 for x in range(self.X_test.shape[0])]
    accuracy_macro_train, accuracy_0_train, accuracy_1_train = get_accuracies(self.y_train_full, one_predicion_train)
    accuracy_macro_test, accuracy_0_test, accuracy_1_test = get_accuracies(self.y_test, one_predicion_test)


  def _get_baseline_increment(self, PERIODS):
    df = self.data_origin.copy()
    #df['mean'] = df[self.COL_STATS].rolling(7).mean()
    df['shift'] = df[self.COL_STATS].shift(PERIODS)
    df.dropna(inplace=True)
    df['pred_mean'] = np.where(df[self.COL_STATS] > df['shift'], 1, 0)
    df = df[[self.COL_STATS, 'pred', 'shift', 'pred_mean']]
    #print(df[500:520])
    v_pred_mean = df['pred_mean'].tolist()
    idx_train = int(len(v_pred_mean) * self.TRAIN_SIZE)
    y_train = v_pred_mean[:idx_train]
    y_test = v_pred_mean[idx_train:]
    v_ytrue = df['pred'].tolist()
    y_true_train = v_ytrue[:idx_train]
    y_true_test  = v_ytrue[idx_train:]
    
    accuracy_macro_train, accuracy_0_train, accuracy_1_train = get_accuracies(y_true_train, y_train, 'TRAIN')
    accuracy_macro_test, accuracy_0_test, accuracy_1_test = get_accuracies(y_true_test, y_test, 'TEST ')


    

  def _train_random_forest(self, do_grid_search=None, X_train=None, y_train=None, X_test=None, y_test=None, print_results=None, save_self=None):
    """ Function that uses random forest classifier to train the model"""
    X_train = self.X_train_full if X_train is None else X_train
    y_train = self.y_train_full if y_train is None else y_train
    X_test = self.X_test if X_test is None else X_test
    y_test = self.y_test if y_test is None else y_test
    do_grid_search = True if do_grid_search is True else False

    
    # Create a new random forest classifier
    if do_grid_search == False:
      rf = RandomForestClassifier(n_jobs=-1, n_estimators=200, random_state=65)
      rf.fit(X_train, y_train)
      rf_best = rf
    else:
      rf = RandomForestClassifier() # # rf = RandomForestClassifier(n_jobs=-1, n_estimators=85, random_state=65)
      
      # Dictionary of all values we want to test for n_estimators
      L = X_train.shape[0]
      if L >= 500:
        params_rf = {'n_estimators': [110,130,140,150,160,180,200]}
      elif L < 500 and L > 100:
        params_rf = {'n_estimators': [20,50,100,150,200,250]}
      else:
        params_rf = {'n_estimators': [5,10,20,30,40,50]}
        params_rf['n_estimators'] = [x for x in params_rf['n_estimators'] if x < L/2]
      
      # Use gridsearch to test all values for n_estimators
      #rf_gs = GridSearchCV(rf, params_rf, cv=5, n_jobs=-1)
      rf_gs = GridSearchCV(rf, params_rf, scoring='accuracy', cv=TimeSeriesSplit().split(X_train)) # scoring='neg_mean_absolute_error'
              
      # Fit model to training data
      rf_gs.fit(X_train, y_train)
      
      # Save best model
      rf_best = rf_gs.best_estimator_
      if print_results == True:
        print('\n RF Best n_estimators Value', rf_gs.best_params_)


    print(' RF TRAIN cross_val_score ==> ', round(cross_val_score(rf_best, X_train, y_train, cv=5, scoring='accuracy').mean(),2))
    print(' RF TEST  cross_val_score ==> ', round(cross_val_score(rf_best, X_test, y_test, cv=5, scoring='accuracy').mean(),2))

    print(' RF TRAIN cross_val_score ts_split  ==> ', round(cross_val_score(rf_best, X_train, y_train, cv=TimeSeriesSplit().split(X_train), scoring='accuracy').mean(),2))
    cv_test = cross_val_score(rf_best, X_test, y_test, cv=TimeSeriesSplit().split(X_test), scoring='accuracy')
    print(' RF TEST  cross_val_score ts_split  ==> ', round(cv_test.mean(),2))
    print('cv_test ==>', cv_test)

    # Predictions
    rf_best_prediction = rf_best.predict(X_test)

    print('X_train.shape', X_train.shape)
    print('len(y_test)', len(y_test), '     len(rf_best_prediction)', len(rf_best_prediction))
    if print_results == True:
      print('\n RF Classification Report \n', classification_report(y_test, rf_best_prediction))
      #print('\n RF Confusion Matrix \n', confusion_matrix(y_test, rf_best_prediction))
      self.df_features_importance = pd.DataFrame({'Feature Importance': rf_best.feature_importances_}, index=self.data.columns[:-1]).sort_values(by='Feature Importance', ascending=False)
      #print(self.df_features_importance)
      #for i in range(len(self.rf_best.feature_importances_)):#  print('      ', round(self.rf_best.feature_importances_[i],4), ' <= ', self.data.columns[i])
    if save_self == True:
      self.rf_best = rf_best
      self.rf_best_prediction = rf_best_prediction
    else:
      return rf_best




  def plot_RF_learning_curve(self):
    from sklearn.model_selection import learning_curve

    OPT = [None, 2, 10]
    OPT = [50,200,500]
    opt_label = 'n_estimators'
    #OPT = [None]
    train_sizes, train_means, test_means, test_stds, train_stds = [],[],[],[],[]
    for opt in OPT:
      dt_mlf = RandomForestClassifier(n_estimators=opt, max_depth=2)#, n_jobs=-1)  # n_estimators=250, max_leaf_nodes=64  max_features=10)   # min_samples_leaf=mlf, random_state=42)
      train_size, train_scores, test_scores = learning_curve(dt_mlf,
                                                          X=self.X_train_full,
                                                          y=self.y_train_full,
                                                          train_sizes=np.linspace(0.1, 1, 10),
                                                          cv=TimeSeriesSplit().split(self.X_train_full))
                                                          #,n_jobs=-1)
      print('fatto {}'.format(str(opt)))
      train_means.append(np.mean(train_scores, axis=1))
      train_stds.append(np.std(train_scores, axis=1))
      test_means.append(np.mean(test_scores, axis=1))
      test_stds.append(np.std(test_scores, axis=1))
      train_sizes.append(train_size)

    fig= plt.figure(figsize=(12, 8))
    for i in range(len(OPT)):
      ax = fig.add_subplot(231+i)
      ax.plot(train_sizes[i], train_means[i], color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes[i], train_means[i] + train_stds[i], train_means[i] - train_stds[i], alpha=0.15, color='blue')
      ax.plot(train_sizes[i], test_means[i], color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes[i], test_means[i] + test_stds[i], test_means[i] - test_stds[i], alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.4,1))
      ax.set_ylabel('Accuracy')
      ax.legend(loc='lower right')
      ax.set_title(r"{}: {}".format(opt_label, OPT[i]), fontsize=12)
    plt.show()






  def _train_KNN(self, X_train=None, do_grid_search=None, y_train=None, X_test=None, y_test=None, print_results=None, save_self=None):

    X_train = self.X_train_full if X_train is None else X_train
    y_train = self.y_train_full if y_train is None else y_train
    X_test = self.X_test if X_test is None else X_test
    y_test = self.y_test if y_test is None else y_test
    do_grid_search = True if do_grid_search is True else False

    if do_grid_search == False:
      knn = KNeighborsClassifier(n_neighbors=1)
      knn.fit(X_train, y_train)
      knn_best = knn
    else:
      knn = KNeighborsClassifier()
      # Create a dictionary of all values we want to test for n_neighbors
      maxN = min(X_train.shape[0]//3, 25)
      params_knn = {'n_neighbors': np.arange(1, maxN)}
      
      # Use gridsearch to test all values for n_neighbors
      #knn_gs = GridSearchCV(knn, params_knn, cv=5)
      knn_gs = GridSearchCV(knn, params_knn, scoring='accuracy',  cv=TimeSeriesSplit().split(X_train)) # scoring='neg_mean_absolute_error'

      # Fit model to training data
      knn_gs.fit(X_train, y_train)
      # Save best model
      knn_best = knn_gs.best_estimator_

      if print_results == True:
        print('\n KNN Best Param =>', knn_gs.best_params_)

    # Predictions
    knn_best_prediction = knn_best.predict(X_test)

    if print_results == True:
      print('\n KNN Classification Report \n', classification_report(y_test, knn_best_prediction))
      print('\n KNN Confusion Matrix \n', confusion_matrix(y_test, knn_best_prediction))
    if save_self == True:
      self.knn_best = knn_best
      self.knn_best_prediction = knn_best_prediction
    else:
      return knn_best




  def _train_gradient_boosting(self, do_grid_search=None, X_train=None, y_train=None, X_test=None, y_test=None, print_results=None, save_self=None):
    X_train = self.X_train_full if X_train is None else X_train
    y_train = self.y_train_full if y_train is None else y_train
    X_test = self.X_test if X_test is None else X_test
    y_test = self.y_test if y_test is None else y_test
    do_grid_search = True if do_grid_search is True else False

    params = {'n_estimators': [200], 'max_depth': [10]} # provate combo con 250,500 |  10,20 ... migliore 250, 10 --- ma top 200,10


    ''' SINGOLO
    '''
    if do_grid_search == False:
      gBoost = GradientBoostingClassifier(n_estimators=200, max_depth=10)
      gBoost.fit(X_train, y_train)
      gBoost_best = gBoost
    else:
      gBoost = GradientBoostingClassifier()
      gBoost_gs = GridSearchCV(gBoost, params, scoring='accuracy',  cv=TimeSeriesSplit().split(X_train), verbose=4) # scoring='neg_mean_absolute_error'
      gBoost_gs.fit(X_train, y_train)
      gBoost_best = gBoost_gs.best_estimator_
      if print_results == True:
        print('\n gBoost Best Param =>', gBoost_gs.best_params_)

    gBoost_best_prediction = gBoost_best.predict(X_test)


    '''
       gBoost_best GradientBoostingClassifier(max_depth=10, n_estimators=200)
            gBoost Classification Report
                                precision   recall  f1-score   support

                        0.0        0.88      0.68      0.77       522
                        1.0        0.84      0.95      0.89       954

                    accuracy                           0.85      1476
                  macro avg        0.86      0.81      0.83      1476
                weighted avg       0.86      0.85      0.85      1476


            gBoost Confusion Matrix
                [[353 169]
                [ 46 908]]

      .............................................................................
            gBoost_best GradientBoostingClassifier(max_depth=10, n_estimators=250)
            gBoost Classification Report
                          precision    recall  f1-score   support

                    0.0       0.87      0.70      0.77       536
                    1.0       0.84      0.94      0.89       940

                accuracy                          0.85      1476
              macro avg       0.86      0.82      0.83      1476
            weighted avg      0.85      0.85      0.85      1476


            gBoost Confusion Matrix
            [[373 163]
            [ 56 884]]
    '''

    # calculate error on validation set



    if print_results == True:
      print('\n gBoost Classification Report \n', classification_report(y_test, gBoost_best_prediction))
      print('\n gBoost Confusion Matrix \n', confusion_matrix(y_test, gBoost_best_prediction))
    if save_self == True:
      self.gBoost_best = gBoost_best
      self.gBoost_best_prediction = gBoost_best_prediction
    else:
      return gBoost_best





  def _train_ensemble_model(self, estimators=None, X_train=None, y_train=None, X_test=None, y_test=None, print_results=None, save_self=None):
    
    X_train = self.X_train_full if X_train is None else X_train
    y_train = self.y_train_full if y_train is None else y_train
    X_test = self.X_test if X_test is None else X_test
    y_test = self.y_test if y_test is None else y_test

    # Create a dictionary of our models
    estimators=[('rf', self.rf_best), ('knn', self.knn_best)] if estimators is None else estimators
    print('    estimators ===>', estimators)
    
    # Create our voting classifier, inputting our models
    ens = VotingClassifier(estimators, voting='hard')
    
    #fit model to training data
    ens.fit(X_train, y_train)
    
    # Predictions
    ens_prediction = ens.predict(X_test)

    #print('TRAIN ENS', round(cross_val_score(ens, self.X_train_full, self.y_train_full, cv=10, scoring='accuracy').mean(),2))

    if print_results == True:
      print('\n Ensemble Score (Test data) =>', round(ens.score(X_test, y_test),2))
      print('\n ENS Classification Report \n', classification_report(y_test, ens_prediction))
      print('\n RF Confusion Matrix \n', confusion_matrix(y_test, ens_prediction))

    if save_self == True:
      self.ens = ens
      self.ens_prediction = ens_prediction
      self.ens_score = round(ens.score(X_test, y_test),2)
    else:
      return ens



  def _train_M1(self, soglia=None, X_train=None, y_train=None, X_test=None, y_test=None, X_valid=None, y_valid=None, print_results=None, save_self=None):
    X_train = self.X_train_full if X_train is None else X_train
    y_train = self.y_train_full if y_train is None else y_train
    X_test = self.X_test if X_test is None else X_test
    y_test = self.y_test if y_test is None else y_test
    #X_valid = self.X_valid if X_valid is None else X_valid
    #y_valid = self.y_valid if y_valid is None else y_valid
    soglia = 0.5 if soglia is None else soglia

    M1 = Sequential()
    input_shape = X_train.shape[1:]

    M1 = Sequential(
      [
        Input(shape=input_shape, name="input"),
        Dense(1024, activation="relu", name="layer1"),
        Dense(512, activation="relu", name="layer2"),
        Dense(256, activation="relu", name="layer3"),
        Dense(128, activation="relu", name="layer4"),
        Dense(48, activation="relu", name="layer5"),#, input_dim=14),# input_shape=input_shape),
        Dense(16, activation="relu", name="layer6"),
        Dense(1, activation='linear', name="output") #  activation= 'linear' 'sigmoid' 'relu'   |   , kernel_initializer='random_normal' ==> Random normal initializer generates tensors with a normal distribution. For uniform distribution, we can use Random uniform initializers.
      ]
    )
    #print(M1.summary())
    M1.compile(loss='mse', optimizer='adam', metrics=['accuracy']) # loss = 'mse' | 'sparse_categorical_crossentropy' 'binary_crossentropy'
    M1_history = M1.fit(X_train, y_train, epochs=10, validation_split=0.33, batch_size=32)#, verbose=0)  #, batch_size = 64
    #M1_Train_loss, M1_Train_accuracy = mse_train = M1.evaluate(X_train, y_train)
    #M1_Train_loss, M1_Train_accuracy = round(M1_Train_loss,2), round(M1_Train_accuracy,2)

    # TODO plot   M1_prediction_raw
    M1_train_prediction_raw = M1.predict(X_train)
    lib_plot_distribution(M1_train_prediction_raw,'M1_train_prediction_raw')
    #print('M1_prediction_raw', M1_prediction_raw)

    v_soglie = [x/100 for x in range(0,101,5)]
    best_accuracy_train, best_soglia_train = 0, 0
    best_accuracy_test, best_soglia_test = 0, 0
    v_accuracy_train, v_accuracy_test = [], []

    for soglia in v_soglie:

      M1_train_prediction_raw = M1.predict(X_train)
      M1_train_prediction = (M1_train_prediction_raw > soglia)
      M1_train_prediction = [1 if x == True else 0 for x in M1_train_prediction]

      M1_test_prediction_raw = M1.predict(X_test)
      M1_test_prediction = (M1_test_prediction_raw > soglia)
      M1_test_prediction = [1 if x == True else 0 for x in M1_test_prediction]
      #print('M1_prediction > soglia ', M1_prediction)

      #print('M1_prediction', M1_prediction)
      #print('len(M1_prediction)', len(M1_prediction), '    sum(M1_prediction)', sum(M1_prediction))
    
      #M1_Test_loss, M1_Test_accuracy = mse_test = M1.evaluate(X_test, y_test, verbose=0)M1_Test_loss, M1_Test_accuracy = round(M1_Test_loss,2), round(M1_Test_accuracy,2)
      M1_Train_accuracy = round(metrics.accuracy_score(y_train, M1_train_prediction),2)
      M1_Test_accuracy = round(metrics.accuracy_score(y_test, M1_test_prediction),2)

      v_accuracy_train.append(M1_Train_accuracy)
      v_accuracy_test.append(M1_Test_accuracy)

      if M1_Train_accuracy > best_accuracy_train:
        best_accuracy_train = M1_Train_accuracy
        best_soglia_train = soglia
      if M1_Test_accuracy > best_accuracy_test:
        best_accuracy_test = M1_Test_accuracy
        best_soglia_test = soglia

      if print_results == True:
        print('  SOGLIA:', soglia, '     M1_Train_accuracy', M1_Train_accuracy, '    M1_Test_accuracy', M1_Test_accuracy)
        #print('\n M1_Train_loss', M1_Train_loss, '    M1_Train_accuracy', M1_Train_accuracy, '  |  M1_Test_loss', M1_Test_loss, '    M1_Test_accuracy', M1_Test_accuracy)
        #print('\n M1 Ensemble Score (Test data) =>', round(M1.score(X_test, y_test),2)) NON ESISTE  --->   AttributeError: 'Sequential' object has no attribute 'score'
        #print('\n M1 Classification Report \n', classification_report(y_test, M1_prediction))
        #print('\n M1 Confusion Matrix \n', confusion_matrix(y_test, M1_prediction))
    print(' BEST ==> |  TRAIN:   best_soglia_train:', best_soglia_train, '     best_accuracy_train:', best_accuracy_train)
    print(' BEST ==> |  TEST:     best_soglia_test:',  best_soglia_test, '      best_accuracy_test:', best_accuracy_test)
    #lib_plot_distribution(v_accuracy_train, 'v_accuracy_train')
    #lib_plot_distribution(v_accuracy_test, 'v_accuracy_test')
    lib_plot_line(v_soglie, v_accuracy_train, 'v_accuracy_train')
    lib_plot_line(v_soglie, v_accuracy_test, 'v_accuracy_test')
    if save_self == True:
      self.M1 = M1
      self.M1_prediction = M1_train_prediction
      self.M1_history = M1_history
      self.M1_Train_accuracy = M1_Train_accuracy
      self.M1_Test_accuracy = M1_Test_accuracy
    else:
      return M1

    #x1 = self.X_train.iloc[10:11].values
    #y = M1(x1)
    #print('    y ->', y)
    #layer.weights  # Empty   | W_h1, b_h1 = model.layers[1].get_weights()
    # history.params, history.epoch  ===> ({'verbose': 1, 'epochs': 5, 'steps': 860}, [0, 1, 2, 3, 4])
    # pd.DataFrame(history.history).plot(figsize=(12,4)); plt.grid(True)



  def _get_dummy_classifiers(self):
    '''
    - Classificatori Banali
      DummyClassifier:  is a classifier that makes predictions using simple rules.
                        This classifier is useful as a simple baseline to compare with other (real) classifiers. Do not use it for real problems.
              mf_dum_cls = DummyClassifier(strategy='most_frequent')
              uni_dum_cls = DummyClassifier(strategy='uniform')
              st_dum_cls = DummyClassifier(strategy='stratified')

              X_train, X_test, y_train, y_test = train_test_split(feature_matrix, credit_card_label, test_size=0.3, random_state=45)
              print('most_frequent', cross_val_score(mf_dum_cls, X_train, y_train, cv=10, scoring='accuracy').mean())
              print('uniform', cross_val_score(uni_dum_cls, X_train, y_train, cv=10, scoring='accuracy').mean())
              print('stratified', cross_val_score(st_dum_cls, X_train, y_train, cv=10, scoring='accuracy').mean())
    '''

    mf_dum_cls = DummyClassifier(strategy='most_frequent')
    uni_dum_cls = DummyClassifier(strategy='uniform')
    st_dum_cls = DummyClassifier(strategy='stratified')

    print('TRAIN most_frequent', round(cross_val_score(mf_dum_cls, self.X_train_full, self.y_train_full, cv=10, scoring='accuracy').mean(),2))
    print('TRAIN uniform', round(cross_val_score(uni_dum_cls, self.X_train_full, self.y_train_full, cv=10, scoring='accuracy').mean(),2))
    print('TRAIN stratified', round(cross_val_score(st_dum_cls, self.X_train_full, self.y_train_full, cv=10, scoring='accuracy').mean(),2))

    print('TEST  most_frequent', round(cross_val_score(mf_dum_cls, self.X_test, self.y_test, cv=10, scoring='accuracy').mean(),2))
    print('TEST  uniform', round(cross_val_score(uni_dum_cls, self.X_test, self.y_test, cv=10, scoring='accuracy').mean(),2))
    print('TEST  stratified', round(cross_val_score(st_dum_cls, self.X_test, self.y_test, cv=10, scoring='accuracy').mean(),2))


  def _data__clean__indicators__split(self, prediction_window=15):

    t1 = time.time()
    self._produce_data(prediction_window=prediction_window)
    self._split_data()
    print(str(round(time.time() - t1,2)) + ' seconds to clean data')



  def _models_fit_predict(self):
    '''
    Aggiungi modelli basici baseline:
      - sempre 1      DONE
      - MM 7g         DONE
      - proporzionale
      - ARIMA
    '''
    
    #self._train_baseline_one()
    #self._get_baseline_increment(7)
    #self._get_dummy_classifiers()
    
    
    t1 = time.time()
    #self.plot_RF_learning_curve()
    self._train_random_forest(print_results=True, save_self=True) #   self.rf_best  |   self.rf_best_prediction    |    self.df_features_importance
    print(str(round(time.time() - t1,2)) + ' seconds for RF')

    '''

    t1 = time.time()
    self._train_KNN(print_results=True, save_self=True)           #   knn_best   |     knn_best_prediction
    print(str(round(time.time() - t1,2)) + ' seconds for KNN')


    t1 = time.time()
    self._train_gradient_boosting(print_results=True, save_self=True)  #       self.gBoost_best   |   self.gBoost_best_prediction
    print(str(round(time.time() - t1,2)) + ' seconds for GBC')

    t1 = time.time()
    estimators = [('rf', self.rf_best), ('knn', self.knn_best), ('gBoost', self.gBoost_best)]
    self._train_ensemble_model(estimators, print_results=True, save_self=True)      #  ens  |   ens_prediction  |  ens_score
    print(str(round(time.time() - t1,2)) + ' seconds for Ensemble Model')

    

    t1 = time.time()
    self._train_M1(print_results=True, save_self=True) 
    print(str(round(time.time() - t1,2)) + ' seconds for M1')
    
    '''




PREDICTION_WINDOW = 10
t = Ticker('SPY')
t._data__clean__indicators__split(prediction_window=PREDICTION_WINDOW)
print(t.data)

#t.data.to_excel("output3.xlsx")  

#t._models_fit_predict()
#t.cross_Validation()


'''
TODO:
- Zignani ==> Soglia nel Neural Network
- Classificatori Banali
  DummyClassifier:  is a classifier that makes predictions using simple rules.
                    This classifier is useful as a simple baseline to compare with other (real) classifiers. Do not use it for real problems.
          mf_dum_cls = DummyClassifier(strategy='most_frequent')
          uni_dum_cls = DummyClassifier(strategy='uniform')
          st_dum_cls = DummyClassifier(strategy='stratified')

          X_train, X_test, y_train, y_test = train_test_split(feature_matrix, credit_card_label, test_size=0.3, random_state=45)
          print('most_frequent', cross_val_score(mf_dum_cls, X_train, y_train, cv=10, scoring='accuracy').mean())
          print('uniform', cross_val_score(uni_dum_cls, X_train, y_train, cv=10, scoring='accuracy').mean())
          print('stratified', cross_val_score(st_dum_cls, X_train, y_train, cv=10, scoring='accuracy').mean())
- XGBoost
- LSTM, GRE


DOMANDE:
- Perchè questi parametri? .... Grid Search ....   Unità di Neuroni .... 
- Non sono troppi alberi per TOT (se pochi) dati
- Perchè la Grid Search su quel parametro?
- Perchè KNN e non K-means?


'''



