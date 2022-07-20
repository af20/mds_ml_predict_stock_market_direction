import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, StratifiedKFold
from input_values import *

def lib_get_ind_or_ticker_from_id(id, ind_or_ticker):
  if ind_or_ticker == 'ticker':
    from input_values import v_d_tickers as v_some
  elif ind_or_ticker == 'macro_ind':
    from input_values import v_d_macro_indicators as v_some
  elif ind_or_ticker == 'tech_ind':
    from input_values import v_d_tech_indicators as v_some
  else:
    raise Exception('Wrong input value for => ind_or_ticker')

  try:
    x = [y for y in v_some if y['id'] == id][0]
    return x
  except:
    return None




def do_Grid_Search(M, grid, X, Y, file_name_base, n_iter=None, n_splits=None):
  def get_file_path(N):
    return 'results/'+str(ticker_to_predict) + '_' + str(PERIODS_TO_FORECAST) + '_' +file_name_base+'_' + str(N)+".xlsx"

  n_iter = 20 if n_iter is None else n_iter
  n_splits = N_FOLDS# 5 if n_splits is None else n_splits

  if CV_MODE == 'TS':
    CV = TimeSeriesSplit(n_splits=n_splits).split(X)
  elif CV_MODE == 'SKF':
    CV = StratifiedKFold(n_splits=n_splits).split(X,Y)

  #import tensorflow as tf
  #from keras import KerasClassifier
  #Kmodel = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=M, verbose=1)

  #M_CV = GridSearchCV(estimator=M, param_grid=grid, cv=TimeSeriesSplit().split(X), return_train_score=True, verbose=2)
  M_CV = RandomizedSearchCV(estimator=M, scoring=['accuracy', 'recall', 'precision', 'f1_macro'], refit='f1_macro', param_distributions=grid, cv=CV, return_train_score=True, n_iter=n_iter, n_jobs=2, verbose=2)
  M_CV.fit(X, Y)

  # best_params = M_CV.best_params_
  df = pd.DataFrame(M_CV.cv_results_)
  print(df)
  # DF v_cols ==> v_split_train = ['split'+str(i)+'_train_score' for i in range(n_splits)] # v_split_test = ['split'+str(i)+'_test_score' for i in range(n_splits)] # v_cols = ['params', 'mean_test_score', 'mean_train_score'] + v_split_train + v_split_test  #df = df[v_cols]

  df['mean_diff_accuracy'] = df['mean_test_accuracy'] - df['mean_train_accuracy']
  df['mean_diff_f1_macro'] = df['mean_test_f1_macro'] - df['mean_train_f1_macro']
  df.sort_values(by=['mean_test_f1_macro'], inplace=True, ascending=False)
  
  N = 1
  PATH_ = get_file_path(N)
  while os.path.exists(PATH_) == True:
    N+=1
    PATH_ = get_file_path(N)

  df.to_excel(PATH_) #df.to_csv(file_name+'.csv')
  return df, M_CV.best_estimator_


def print_accuracy_train_test(y_train, y_test, y_pred_tr, y_pred_te, label=None, y_valid=None, y_pred_valid=None):
  # funzione di fit_predict
  label = '' if label is None else str(label)
  from sklearn import metrics
  M_Train_accuracy = round(metrics.accuracy_score(y_train, y_pred_tr),2)
  M_Test_accuracy = round(metrics.accuracy_score(y_test, y_pred_te),2)
  #M_Valid_accuracy = round(metrics.precision_score(y_train, y_pred_tr),2)

  str_valid = ''
  if y_pred_valid is not None:
    M_Valid_accuracy = round(metrics.accuracy_score(y_valid, y_pred_valid),2)
    str_valid = str('    Valid:' + str(M_Valid_accuracy))

  try:
    pos_perc_tr_real = round(sum([x[0] for x in y_train]) / len(y_train),2)
    pos_perc_te_real = round(sum([x[0] for x in y_test]) / len(y_test),2)
  except:
    pos_perc_tr_real = round(sum(y_train) / len(y_train),2)
    pos_perc_te_real = round(sum(y_test) / len(y_test),2)

  pos_perc_tr = round(sum(y_pred_tr) / len(y_pred_tr),2)
  pos_perc_te = round(sum(y_pred_te) / len(y_pred_te),2)
  print('  ' + label + '  | ACCURACY  Train:', M_Train_accuracy, str_valid,'    Test:', M_Test_accuracy, '    | POS PERC => REAL_TR:', pos_perc_tr_real, '   REAL_TE:', pos_perc_te_real,'   PRED_TR:', pos_perc_tr, ' PRED_TE:', pos_perc_te)





def plot_Learning_Curves(v_opt, label, train_sizes, train_means, test_means, test_stds, train_stds):
  fig= plt.figure(figsize=(12, 8))
  for i in range(len(train_sizes)):
    ax = fig.add_subplot(231+i)
    ax.plot(train_sizes[i], train_means[i], color='blue', marker='o', markersize=5, label='Training accuracy')
    ax.fill_between(train_sizes[i], train_means[i] + train_stds[i], train_means[i] - train_stds[i], alpha=0.15, color='blue')
    ax.plot(train_sizes[i], test_means[i], color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
    ax.fill_between(train_sizes[i], test_means[i] + test_stds[i], test_means[i] - test_stds[i], alpha=0.15, color='green')
    ax.grid()
    ax.set_ylim((0.4, 1))
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.set_title(r"{}: {}".format(label, v_opt[i]), fontsize=12)
  plt.show()


def plot_Validation_Curve(v_param_range, param_name, train_scores, test_scores):
  train_mean = np.mean(train_scores, axis=1); train_std = np.std(train_scores, axis=1); test_mean = np.mean(test_scores, axis=1); test_std = np.std(test_scores, axis=1)
  #print('train_mean', train_mean, '    |   train_std', train_std)
  #print('test_mean', test_mean,'    |   test_std', test_std)
  fig=plt.figure(figsize=(12,7))
  ax = fig.add_subplot()
  ax.plot(v_param_range, train_mean, color='blue', label='Training accuracy')
  ax.plot(v_param_range, test_mean, color='green', label='Validation accuracy')
  try:
    ax.fill_between(v_param_range, train_mean+train_std, train_mean-train_std, alpha=0.15, color='blue')
  except:
    pass
  try:
    ax.fill_between(v_param_range, test_mean+test_std, test_mean-test_std, alpha=0.15, color='green')
  except:
    pass
  ax.grid()
  ax.legend(loc='lower right')
  ax.set_ylabel('Accuracy')
  minY = min(min(train_mean), min(test_mean))*0.9
  ax.set_ylim([0, 1.03])
  ax.set_xlabel(param_name)
  ax.set_ylabel('Accuracy')
  plt.show()



def get_binary_prediction_with_treshold(v_hd, treshold):
  #v_hd_binary = np.where(v_hd >= treshold, v_hd, -v_hd)
  v_hd_binary = [1 if x >= treshold else 0 for x in v_hd]
  return v_hd_binary


def read_from_excel_best_params_Grid_Search(file_name, idx):
  idx = 0 if idx is None else idx  
  df = pd.read_excel(file_name)    
  L = df.shape[0]
  idx = max(0, min(idx, L-1))
  best_params = df.iloc[idx]['params']
  best_params = ast.literal_eval(best_params)
  return best_params


def lib_print_tree(M, idx):
  from sklearn import tree
  tree.plot_tree(M.estimators_[idx])
  plt.show()


def lib_plot_2_lines(v_x, ds1, ds2):
  # ds1 = {'name': [], 'values': []}
  import matplotlib.patches as mpatches
  v_1 = [x for x in ds1['values'] if np.isnan(x) == False]
  v_2 = [x for x in ds2['values'] if np.isnan(x) == False]
  delta = len(v_1) - len(v_2)
  v_1 = v_1[delta:]
  v_x = v_x[delta:]
  
  v_colors = ['blue', 'red']
  fig, ax = plt.subplots(figsize=(14, 8))
  ax2 = ax.twinx()

  ax.plot(v_x, v_1, color=v_colors[0], label=ds1['name'])
  ax2.plot(v_x, v_2, color = v_colors[1], label=ds2['name'])
  v_legend_patches = [
    mpatches.Patch(color=v_colors[0], label=ds1['name']),
    mpatches.Patch(color=v_colors[1], label=ds2['name']),
  ]
  ax.set_yscale('log')
  plt.legend(handles=v_legend_patches, loc='lower right')
  ax.set_ylabel(ds1['name'])
  ax2.set_ylabel(ds2['name'])
  plt.show()
