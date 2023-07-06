from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

import numpy as np

np.random.seed(7) # definindo uma semente global

class ViewModelsResults:
    
    def get_best_params(self, model_name, results):
      """get model best params

      Args:
          model_name (string): name of model
          results (list): list of tuples with the best results

      Returns:
          dict: model best results
      """
      best_model = filter(lambda x: x[0] == model_name, results)
      return list(best_model)[0][2]
    
    
    def __get_bases_for_votting_classifier(self, best_decision_tree, best_svm):
      """get votting classifier best params

      Args:
          best_decision_tree (dict):
          best_svm (dict): 

      Returns:
          list: list of bases for VotingClassifier
      """
      # criando os modelos para o VotingClassifier
      bases = []
      model1 = LogisticRegression(max_iter=200)
      bases.append(('logistic', model1))
      model2 = DecisionTreeClassifier(**best_decision_tree)
      bases.append(('cart', model2))
      model3 = SVC(**best_svm)
      bases.append(('svm', model3))
      return bases
    
    def handle_pipelines(self, best_params_list, tag):
      """create pipelines

      Args:
          best_params_list (list): list with best params 
          tag (string): tag for df

      Returns:
          list: pipelines list
      """
      pipelines = []
      best_knn = self.get_best_params("KNN", best_params_list)
      best_decision_tree = self.get_best_params("Decision Tree", best_params_list)
      best_navie_bayes = self.get_best_params("Navie Bayes", best_params_list)
      best_svm = self.get_best_params("SVM", best_params_list)
      
      
      # definindo os parâmetros do classificador base para o ensambles
      base = DecisionTreeClassifier(**best_decision_tree)
      num_trees = 100
      max_features = 3

      bases = self.__get_bases_for_votting_classifier(best_decision_tree, best_svm)

      # Criando os elementos do pipeline

      # Algoritmos que serão utilizados
      reg_log = ('LR', LogisticRegression(max_iter=200))
      knn = ('KNN', KNeighborsClassifier(**best_knn))
      cart = ('Decision Tree', DecisionTreeClassifier(**best_decision_tree))
      naive_bayes = ('Navie Bayes', GaussianNB(**best_navie_bayes))
      svm = ('SVM', SVC(**best_svm))
      bagging = ('Bag', BaggingClassifier(base_estimator=base, 
                                          n_estimators=num_trees))
      random_forest = ('RF', RandomForestClassifier(n_estimators=num_trees, 
                                                  max_features=max_features))
      extra_trees = ('ET', ExtraTreesClassifier(n_estimators=num_trees, 
                                              max_features=max_features))
      adaboost = ('Ada', AdaBoostClassifier(n_estimators=num_trees))
      gradient_boosting = ('GB', GradientBoostingClassifier(n_estimators=num_trees))
      voting = ('Voting', VotingClassifier(bases))


      # Montando os pipelines

      # Dataset original
      pipelines.append((f'LR{"-" + tag}', Pipeline([reg_log]))) 
      pipelines.append((f'KNN{"-" + tag}', Pipeline([knn])))
      pipelines.append((f'Decision Tree{"-" + tag}', Pipeline([cart])))
      pipelines.append((f'Navie Bayes{"-" + tag}', Pipeline([naive_bayes])))
      pipelines.append((f'SVM{"-" + tag}', Pipeline([svm])))
      pipelines.append((f'Bag{"-" + tag}', Pipeline([bagging])))
      pipelines.append((f'RF{"-" + tag}', Pipeline([random_forest])))
      pipelines.append((f'ET{"-" + tag}', Pipeline([extra_trees])))
      pipelines.append((f'Ada{"-" + tag}', Pipeline([adaboost])))
      pipelines.append((f'GB{"-" + tag}', Pipeline([gradient_boosting])))
      pipelines.append((f'Vot{"-" + tag}', Pipeline([voting])))
      
      return pipelines
      

    def run(self, best_params_list, X_train, y_train, kfold, scoring, tag):
      """get results for each model

      Args:
          best_params_list (list): list of best params for models
          X_train (list): list of train atributes
          y_train (list): list of train results
          kfold (StratifiedKFold): Stratified KFold
          scoring (string): 
          tag (string): tag for df

      Returns:
          (list, list): list with models results and names
      """
      # armazeando os pipelines e os resultados para todas as visões do dataset
      # como sao datasets diferentes, pelos tratamentos, irei definir dois pipelines 
      pipelines = self.handle_pipelines(best_params_list, tag)
      results = []
      names = []

      # Executando os pipelines
      for name, model in pipelines:
          cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
          results.append(cv_results)
          names.append(name)
          msg = "%s: %.3f (%.3f)" % (name, cv_results.mean(), cv_results.std()) 
          print(msg)
          
      return results, names