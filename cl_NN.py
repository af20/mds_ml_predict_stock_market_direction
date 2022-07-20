'''
INIZIO
- Scelgo un modello dal file 'input_nn'
- Cerco il miglior modello (TRAIN) con GridSearch, in cui faccio il CV (5)
- Uso quel modello per la prediction sul TEST => salvo accuracy (train e test)
FINE
'''
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import warnings
warnings.filterwarnings("ignore")

from datetime import date, datetime
import pandas as pd
import numpy as np

from input_nn import *
from lib_preprocessing import DO_PREPROCESSING
from input_values import *
from lib_general_functions import *

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, KFold, StratifiedKFold, cross_val_score, cross_validate, learning_curve
from sklearn import metrics

import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, InputLayer, Embedding
from keras import Input

df, X_, Y_, X_train_, X_test_, y_train_, y_test_, X_train_part_, X_valid_, y_train_part_, y_valid_ = DO_PREPROCESSING()
X_SHAPE = X_.shape
# Y = Y[col_y_true].tolist()
# y_train = y_train[col_y_true].tolist()
# y_test = y_test[col_y_true].tolist()
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

tf.random.set_seed(1234)

class c_NN:
  def __init__(self, d_Model):
    self.d_Model = d_Model
    if d_Model['type'] == 'normal':
      M = KerasClassifier(model=self.create_model(), verbose=0)
    else:
      M = self.create_model()
    M = KerasClassifier(model=self.create_model(), verbose=0)

    #print(self.create_model())

    self.M = M
    self.create_param_grid()


  def create_model(self):
    # , kernel_constraint=MaxNorm(weight_constraint)))
	  #   model.add(Dropout(dropout_rate))
    M = Sequential()

    DM = self.d_Model
    if DM['type'] == 'normal':
      M.add(InputLayer(input_shape=X_SHAPE[1:])) #M.add(InputLayer(input_shape=X_train.shape[1:]))
    elif DM['type'] == 'special_1':
      #M.add(Embedding(X_SHAPE[0], X_SHAPE[1], input_length=500))
      #M.add(Embedding(N°ROWS, embedding_vecor_length=BOH, input_length=BOH))
      M.add(InputLayer(input_shape=(X_SHAPE[1],1))) #X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    elif DM['type'] == 'special_2':
      #M.add(InputLayer(input_shape=(X_SHAPE[1],1,0)))
      M.add(Input(shape=(299, 28, 1)))
      #M.add(Reshape((X_SHAPE[1], 1, 1)))

    for l in DM['v_layers']:
      M.add(l)
    M.add(Dense(1, activation='sigmoid', name="output")) #  activation= 'linear' 'sigmoid' 'relu'   |   , kernel_initializer='random_normal' ==> Random normal initializer generates tensors with a normal distribution. For uniform distribution, we can use Random uniform initializers.
    M.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # loss = 'mse' | 'sparse_categorical_crossentropy' 'binary_crossentropy'
    return M


  def trasform_shape(self, X_train, y_train, X_test=None, y_test=None):
    if self.d_Model['type'] == 'special_1':
      X_train, y_train = np.array(X_train), np.array(y_train) # X_train(2961,60)    y_train(2961,)
      X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # (2961, 60, 1)   work with NN
      if X_test is not None:
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_train, y_train, X_test, y_test


  def adjust_predictions(self, y_pred_tr, y_pred_te, treshold):
    if self.d_Model['type'] == 'special_1':
      treshold_tr = np.mean(y_pred_tr)*0.66 if treshold is None else treshold
      treshold_te = np.mean(y_pred_te)*0.66 if treshold is None else treshold
      y_pred_tr = get_binary_prediction_with_treshold(y_pred_tr, treshold_tr)
      y_pred_te = get_binary_prediction_with_treshold(y_pred_te, treshold_te)
      #print('treshold_tr', treshold_tr, '      treshold_te', treshold_te)
    return y_pred_tr, y_pred_te


  def fit_predict_model(self, M, X_train, y_train, X_test, y_test, treshold=None):
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=1, restore_best_weights=True)    #callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.01)
    X_train, y_train, X_test, y_test = self.trasform_shape(X_train, y_train, X_test, y_test)
    M.fit(X_train, y_train, callbacks=[callback], verbose=0)
    y_pred_tr = M.predict(X_train)
    y_pred_te = M.predict(X_test)
    y_pred_tr, y_pred_te = self.adjust_predictions(y_pred_tr, y_pred_te, treshold)
    M_Train_accuracy = round(metrics.accuracy_score(y_train, y_pred_tr),2)
    M_Test_accuracy = round(metrics.accuracy_score(y_test, y_pred_te),2)
    treshold = '' if treshold is None else str(treshold)

    y_train = y_train[col_y_true].tolist()
    y_test = y_test[col_y_true].tolist()
    y_pred_tr = [x[0] for x in y_pred_tr]
    y_pred_te = [x[0] for x in y_pred_te]
    print_accuracy_train_test(y_train, y_test, y_pred_tr, y_pred_te, treshold)
    return

    print('y_test ==> ', y_test)

    pos_perc_tr_real = round(sum([x[0] for x in y_train]) / len(y_train),2)
    pos_perc_te_real = round(sum([x[0] for x in y_test]) / len(y_test),2)
    pos_perc_tr = round(sum(y_pred_tr) / len(y_pred_tr),2)
    pos_perc_te = round(sum(y_pred_te) / len(y_pred_te),2)
    print('  ' + treshold + '   Accuracy_train:', M_Train_accuracy, '    Accuracy_test:', M_Test_accuracy, '    | pos_perc => REAL_TR:', pos_perc_tr_real, '  | => REAL_TE:', pos_perc_te_real,' |  TR:', pos_perc_tr, ' | TE:', pos_perc_te)



  def do_kfold_cross_validation(self, M, X, Y):
    X, Y, a, b = self.trasform_shape(X, Y)
    
    # Define the K-fold Cross Validator
    tscv = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=5, test_size=None)
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []

    for train_index, test_index in tscv.split(X):
      if isinstance(X, pd.DataFrame):
        X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
        y_tr, y_te = Y.iloc[train_index], Y.iloc[test_index]
      else:
        X_tr, X_te = X[train_index], X[test_index]
        y_tr, y_te = Y[train_index], Y[test_index]
      
      M = self.create_model(X_tr.shape)
      M_history = M.fit(X_tr, y_tr, epochs=30, validation_split=0.33, batch_size=32, verbose=0)#, verbose=0)  #, batch_size = 64

      # Generate generalization metrics
      scores = M.evaluate(X_te, y_te, verbose=0)
      M_Train_loss, M_Train_accuracy = M.evaluate(X_tr, y_tr, verbose=0)
      M_Test_loss, M_Test_accuracy = M.evaluate(X_te, y_te, verbose=0)

      print(f'Score for fold {fold_no} => TRAIN: {round(M_Train_accuracy,2)}  |  TEST: {round(M_Test_accuracy,2)}')
      acc_per_fold.append(scores[1] * 100)
      loss_per_fold.append(scores[0])
      fold_no += 1


  def create_param_grid(self):
    learn_rate = [0.001, 0.01, 0.1]#, 0.2, 0.3]
    momentum = [0.0, 0.2, 0.5, 0.9]
    batch_size = [10, 20, 50, 100]
    epochs = [5]
    self.param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer__learning_rate=learn_rate, optimizer__momentum=momentum)


  def compute_grid_search(self, X, Y, n_iter):
    #if self.d_Model['type'] == 'special_1':
    #  KerasClassifier(model=self.create_model(), verbose=0)
    #else:
    M = self.M
    # TODO prova a ottimizzare il n° dei neuroni
    file_name = 'NN' + str(self.d_Model['id'])
    self.df_grid_search, self.best_model = do_Grid_Search(M, self.param_grid, X, Y, file_name, n_iter)
    



  def get_best_model(self, N, idx=None):
    # NN like 1
    file_name = 'results/'+str(ticker_to_predict) + '_' + str(PERIODS_TO_FORECAST) +'_NN'+str(self.d_Model['id'])+'_' + str(N) + '.xlsx'
    best_params = read_from_excel_best_params_Grid_Search(file_name, idx)
    self.best_model = KerasClassifier(model=self.create_model(), **best_params)
    #print('    best_model =>', self.best_model)
    

print(df)

M = c_NN(d_M0)
M.get_best_model(N=1, idx=0)
M.fit_predict_model(M.M, X_train_, y_train_, X_test_, y_test_)
o=827/0
for i in range(10):
  M.get_best_model(N=1, idx=i)
  M.fit_predict_model(M.best_model, X_train_, y_train_, X_test_, y_test_, i)
a=987/0
M.compute_grid_search(X_, Y_, n_iter=10)
a=987/0


v_tresh = [x/1000 for x in range(450,600,1)]
v_tresh = [x/100 for x in range(0,100,1)]
for tresh in v_tresh:
  M.fit_predict_model(M.M, X_train_, y_train_, X_test_, y_test_, tresh)
p=987/0
#M.fit_predict_model(M.M, X_train_, y_train_, X_test_, y_test_)
a=98/0
#M.fit_predict_model(M.best_model, X_train_, y_train_, X_test_, y_test_)

#M.do_kfold_cross_validation(M.M, X_, Y_)

