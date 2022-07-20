'''
INIZIO
- Cerco il miglior modello (TRAIN) con GridSearch, in cui faccio il CV (5)
- Uso quel modello per la prediction sul TEST => salvo accuracy (train e test)
FINE
'''
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from datetime import date, datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, KFold, StratifiedKFold, cross_val_score, cross_validate, learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn import metrics


from lib_preprocessing import DO_PREPROCESSING
from input_values import *
from lib_general_functions import *


df, X_, Y_, X_train_, X_test_, y_train_, y_test_, X_train_part_, X_valid_, y_train_part_, y_valid_ = DO_PREPROCESSING()
Y_ = Y_[col_y_true].tolist()
y_train_ = y_train_[col_y_true].tolist()
y_train_part_ = y_train_part_[col_y_true].tolist()
y_valid_ = y_valid_[col_y_true].tolist()
y_test_origin = y_test_.copy()
y_test_ = y_test_[col_y_true].tolist()

'''
  Devo trovare il Modello che generalizza meglio, sarà anche quello con performance più alte OS?
'''


class c_Random_Forest:
  def __init__(self):
    self.M = RandomForestClassifier()#criterion='entropy', max_depth=2, max_features='sqrt', n_estimators=300, random_state=18, bootstrap=True, oob_score=True)
    self.create_param_grid()

  def fit_predict_model(self, M1, X_train, y_train, X_test, y_test, X_valid=None, y_valid=None, label='', print_future_prediction=None):
    # Accuracy of Train & Test
    M1.fit(X_train, y_train)
    y_pred_train, y_pred_test = M1.predict(X_train), M1.predict(X_test)
    if y_valid is not None:
      y_pred_valid = M1.predict(X_valid)
    else:
      y_pred_valid = None

    if print_future_prediction:
      df_pred = pd.DataFrame({'time': y_test_origin.index, 'pred': y_pred_test})
      print('df_pred\n', df_pred[-12:])
    
    print_accuracy_train_test(y_train, y_test, y_pred_train, y_pred_test, label, y_valid, y_pred_valid)

    # Accuracy (Rolling)
    #CV_score = cross_val_score(M, X, Y, cv=TimeSeriesSplit().split(X), scoring='accuracy')
    #print('  RF Rolling Accuracy (All) ==> ', CV_score, '     ', round(CV_score.mean(),2))


  def feature_importance(self, M, X):
    importances = M.feature_importances_    
    # Sort the feature importance in descending order
    sorted_indices = np.argsort(importances)[::-1]
    print(sorted_indices)
    import matplotlib.pyplot as plt
    plt.title('Feature Importance')
    plt.bar(range(X.shape[1]), importances[sorted_indices], align='center')
    plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=90)
    plt.tight_layout()
    plt.show()


  def do_learning_curves(self, X, Y):
    if CV_MODE == 'TS':
      CV = TimeSeriesSplit().split(X)
    elif CV_MODE == 'SKF':
      CV = StratifiedKFold().split(X,Y)

    v_opt = [50,150,300]
    train_sizes, train_means, test_means, test_stds, train_stds = [],[],[],[],[]
    for opt in v_opt:
      #M = RandomForestClassifier(criterion='entropy', max_depth=3, max_features='sqrt', n_estimators=opt, random_state=18)#, bootstrap=True, oob_score=True)
      M = RandomForestClassifier(criterion='entropy', max_depth=2, max_features='sqrt', n_estimators=opt, random_state=18, bootstrap=True, oob_score=True)
      train_size, train_scores, test_scores = learning_curve(M, X=X, y=Y, cv=CV)
      print('fatto {}'.format(str(opt)))
      train_means.append(np.mean(train_scores, axis=1)); train_stds.append(np.std(train_scores, axis=1)); test_means.append(np.mean(test_scores, axis=1)); test_stds.append(np.std(test_scores, axis=1)); train_sizes.append(train_size)
    plot_Learning_Curves(v_opt, 'max_depth', train_sizes, train_means, test_means, test_stds, train_stds)



  def do_validation_curve(self, X, Y, v_param_range: list, param_name: str):
    if CV_MODE == 'TS':
      CV = TimeSeriesSplit().split(X)
    elif CV_MODE == 'SKF':
      CV = StratifiedKFold().split(X,Y)

    train_scores, test_scores = validation_curve(RandomForestClassifier(max_depth=1), X=X, y=Y, param_range=v_param_range, param_name=param_name,cv=CV, n_jobs=2)
    plot_Validation_Curve(v_param_range, param_name, train_scores, test_scores)


  def do_validation_curve_on_param_grid(self, M, X, Y):
    for k,v in self.grid.items():
      print('VC =>', k, v)
      self.do_validation_curve(X, Y, v, k)



  def do_cross_validation(self, X, Y):
    if CV_MODE == 'TS':
      CV = TimeSeriesSplit().split(X)
    elif CV_MODE == 'SKF':
      CV = StratifiedKFold().split(X,Y)

    # Scores (qui c'è tutto, rolling TRAIN, TEST)
    #M = RandomForestClassifier(criterion='entropy', max_depth=2, max_features='sqrt', n_estimators=300, random_state=18, bootstrap=True, oob_score=True)
    scores = cross_validate(M, X, Y, cv=CV, return_estimator = True, return_train_score= True, scoring = ['accuracy'], verbose=1) # (dict) keys => ['estimator', 'train_accuracy', 'test_accuracy', 'fit_time', 'score_time']
    #print(scores)
    self.cross_validation_scores = scores
    
    v_tr = scores['train_accuracy']
    v_te = scores['test_accuracy']
    v_x = [i for i in range(1, len(v_tr)+1)]
    #print('v_tr', v_tr); print('v_te', v_te)
    plt.plot(v_x, v_tr, label='Train Accuracy')
    plt.plot(v_x, v_te, label='Test Accuracy')
    plt.legend(loc="upper left")
    # TODO Aggiungi etichette label (Y: Accuracy) (X: CV test N° ... togli 0.5, 1.5)
    plt.show()


  def create_param_grid(self):
    previous_grid = { 'n_estimators': [200],  'max_depth' : [2,3],
    #'max_features': ['sqrt', 'log2'],    #'criterion' :['gini', 'entropy'], #, 'max_samples: [5,10]
    }
    n_estimators = [10,100,200,500]# [int(x) for x in np.linspace(start = 20, stop = 220, num = 5)] # Number of trees in random forest
    max_features = [3,6,10,'sqrt']#, auto,'sqrt'] # Number of features to consider at every split ..... max_features='auto' means that when building each node, only the square root of the number of features in your training data will be considered to pick the cutoff point that reduces the gini impurity most.
    max_depth = [1,2,3,4,5,10]# [None] + [1, 2, 3] # Maximum number of levels in tree
    min_samples_split = [2, 5, 10, 20, 40] # Minimum number of samples required to split a node
    min_samples_leaf = [1, 2, 4, 8, 16] # Minimum number of samples required at each leaf node
    bootstrap = [True, False] # Method of selecting samples for training each tree ....bootstrap=True means that each tree will be trained on a random sample (with replacement) of a certain percentage of the observations from the training dataset.

    # Create the random grid
    self.grid =  {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'bootstrap': bootstrap
                  }


  def compute_grid_search(self, X, Y, n_iter):
    self.df_grid_search, self.best_model = do_Grid_Search(RandomForestClassifier(), self.grid, X, Y, 'RF', n_iter) # random_state=71
    #print('self.best_model', self.best_model)
    #self.best_params = self.df_grid_search.iloc[0]['params']
    #print('self.best_params', self.best_params)
    #self.best_model_me = RandomForestClassifier(**self.best_params)
    

  def get_best_model(self, N, idx=None, random_state=None):
    random_state=1 if random_state is None else random_state
    file_name = 'results/'+str(ticker_to_predict) + '_' + str(PERIODS_TO_FORECAST) + '_RF_' + str(N) + '.xlsx'
    best_params = read_from_excel_best_params_Grid_Search(file_name, idx)
    self.best_model = RandomForestClassifier(random_state=random_state, **best_params) # random_state = 71
    #print('    best_model =>', self.best_model)
    # PRINT TREE
    # self.best_model.fit(X_, Y_)
    # lib_print_tree(self.best_model, N)

'''
M = c_Random_Forest()
M.get_best_model(14, idx=0, random_state=71)
M.fit_predict_model(M.best_model, X_train_, y_train_, X_test_, y_test_) 

a=987/0
for i in range(100):
  M.get_best_model(14, idx=0, random_state=i) # 55-81%(85%,94%) |  71-81%(85%,89%)  ---------  9
  M.fit_predict_model(M.best_model, X_train_part_, y_train_part_, X_valid_, y_valid_, X_test_, y_test_, i)
  #M.fit_predict_model(M.best_model, X_train_, y_train_, , i)
o=827/0
M.compute_grid_search(X_train_, y_train_, 100)
#M.do_validation_curve(X_, Y_, list(range(1,10)) , 'max_features')
a=9876/0
#i=71
#M.get_best_model(11, idx=0, random_state=i)
#M.fit_predict_model(M.best_model, X_train_, y_train_, X_test_, y_test_, i) 
# Migliore modello file_11, random_state=71
#M.fit_predict_model(M.best_model, X_, y_train_, X_test_, y_test_, i) 
M.get_best_model(14, idx=0)
a=987/0


print(X_train_)
print(y_train_)
a=9876/0

for i in range(400):
  M.get_best_model(11, idx=0, random_state=i)
  M.fit_predict_model(M.best_model, X_train_, y_train_, X_test_, y_test_, i)
a=987/0
#M.feature_importance()
M.compute_grid_search(X_, Y_, n_iter=10)
M.do_validation_curve_on_param_grid(X_, Y_)

a=987/0

#M.get_best_model(5, idx=0)
#M.fit_predict_model(M.best_model, X_train_, y_train_, X_test_, y_test_)

#M.do_learning_curves(X_, Y_)
#M.do_cross_validation(X, Y)
'''