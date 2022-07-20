# Import the libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def lib_plot_line(v_x, v_y, title=None):
  plt.plot(v_x, v_y)
  if title != None:
    plt.title(title)
  plt.show()

def lib_plot_distribution(v_x, title=None):

  # matplotlib histogram
  plt.hist(v_x, color = 'blue', edgecolor = 'black',
          bins = int(180/5))

  # seaborn histogram
  sns.distplot(v_x, hist=True, kde=False, 
              bins=int(180/5), color = 'blue',
              hist_kws={'edgecolor':'black'})
  # Add labels
  if title != None:
    plt.title(title)
  #plt.xlabel('Delay (min)')
  #plt.ylabel('Flights')
  plt.show()




def get_accuracies(y_true, y_pred, label=None):
  from sklearn.metrics import accuracy_score
  accuracy_macro = round(accuracy_score(y_true, y_pred),2)
  y_true_0_idx = set([i for i,x in enumerate(y_true) if x == 0])
  y_true_1_idx = set([i for i,x in enumerate(y_true) if x == 1])

  y_true_0 = [x for i,x in enumerate(y_true) if i in y_true_0_idx]
  y_pred_0 = [x for i,x in enumerate(y_pred) if i in y_true_0_idx]

  y_true_1 = [x for i,x in enumerate(y_true) if i in y_true_1_idx]
  y_pred_1 = [x for i,x in enumerate(y_pred) if i in y_true_1_idx]

  accuracy_0 = round(accuracy_score(y_true_0, y_pred_0),2)
  accuracy_1 = round(accuracy_score(y_true_1, y_pred_1),2)
  label = label if type(label) == str else ''
  print(' =>', label,' Accuracy | M:', accuracy_macro, '  |  0:', accuracy_0, '  |  1:', accuracy_1)
  return accuracy_macro, accuracy_0, accuracy_1





def get_classification_report_accuracy(y_true, y_pred):
  x = classification_report(y_true, y_pred, output_dict=True)
  d = {
    '0.0': x['0.0'],
    '1.0': x['1.0']
  }
  '''
    from sklearn.metrics import precision_recall_fscore_support as score
    precision,recall,fscore,support=score(self.y_train_full, one_predicion_train, average='macro')
    print('Precision : {}'.format(precision))
    print('Recall    : {}'.format(recall))
    print('F-score   : {}'.format(fscore))
    print('Support   : {}'.format(support))


    #get_classification_report_stats()
    CR_train = classification_report(self.y_train_full, one_predicion_train, output_dict=True)

    print('\n One Classification Report (Train) \n', CR_train)
    print('\n One Classification Report (Test) \n', classification_report(self.y_test, one_predicion_test))


      {
        "0.0": 
        {
          "precision": 0,
          "recall": 0,
          "f1-score": 0,
          "support": 2233
        },
        "1.0": 
        {
          "precision": 0.6219739292364991,
          "recall": 1,
          "f1-score": 0.7669345579793342,
          "support": 3674
        },
        "accuracy": 0.6219739292364991,
        "macro avg": 
        {
          "precision": 0.31098696461824954,
          "recall": 0.5,
          "f1-score": 0.3834672789896671,
          "support": 5907
        },
        "weighted avg": 
        {
          "precision": 0.3868515686498895,
          "recall": 0.6219739292364991,
          "f1-score": 0.4770133004936641,
          "support": 5907
        }
      }
  '''
