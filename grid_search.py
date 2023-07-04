from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np


class GridSearchProcess:
  
  def __get_knn_model_params(self):
    k = [1,3,5,7,9,11,13,15,17,19,21]
    distancias = ["euclidean", "manhattan", "minkowski"]
    param_grid = dict(n_neighbors=k, metric=distancias)
    return param_grid
  
  def __get_decision_tree_params(self):
    criterion = ['gini','entropy']
    max_depth = [3,4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]
    param_grid = {'criterion': criterion,'max_depth': max_depth}
    return param_grid
  
  def __get_SVM_params(self):
    C = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    param_grid = {
      'C': C,
      'kernel': kernel
    }
    return param_grid
  
  def __get_navie_bayes_params(self):
    return {'var_smoothing': np.logspace(0,-9, num=20)}
  
  def run(self, X_train, y_train, scoring, kfold):
    # Processo de GridSearch 

    models = []
    results = []

    #  KNN
    param_grid = self.__get_knn_model_params()
    models.append(("KNN", KNeighborsClassifier(), param_grid))

    # Decision Tree
    param_grid = self.__get_decision_tree_params()
    models.append(("Decision Tree", DecisionTreeClassifier(), param_grid))

    # Navie Bayes 
    param_grid = self.__get_navie_bayes_params()
    models.append(("Navie Bayes", GaussianNB(), param_grid))

    # SVM
    param_grid = self.__get_SVM_params()
    models.append(("SVM", SVC(), param_grid))

    for name, model, param_grid in models:
      grid = GridSearchCV(estimator=model, param_grid=param_grid, 
                        scoring=scoring, cv=kfold)
      grid.fit(X_train, y_train) 
      results.append((name, grid.best_score_, grid.best_params_))
      # imprime o melhor resultado
      print("Melhor %s : %f usando %s" % 
            (name, grid.best_score_, grid.best_params_))
    return results