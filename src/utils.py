import os
import sys
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
import numpy as np
from scipy.sparse import issparse
from src.exception import CustomException

def save_object(file_path, obj):
    try: 
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

from sklearn.metrics import make_scorer, r2_score, accuracy_score, silhouette_score
import logging

def evaluate_models(X_train, y_train, X_test, y_test, models, param, metric, problem_type):
    try:
        model_report = {}
        for model_name, model in models.items():
            if problem_type in ['regression', 'classification']:
                if param[model_name]:  # If there are hyperparameters to tune
                    gs = GridSearchCV(model, param[model_name], cv=5, scoring=make_scorer(metric), n_jobs=-1)
                    gs.fit(X_train, y_train)
                    best_model = gs.best_estimator_
                    logging.info(f"Best model params for {model_name}: {gs.best_params_}")
                else:  # If no hyperparameters to tune (e.g., LinearRegression)
                    best_model = model
                    best_model.fit(X_train, y_train)
                
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)
                train_score = metric(y_train, y_train_pred)
                test_score = metric(y_test, y_test_pred)
                model_report[model_name] = (train_score, test_score)
                logging.info(f"{model_name} - Train Score: {train_score}, Test Score: {test_score}")

            elif problem_type == 'clustering':
                best_score = -np.inf
                best_params = None
                for param_combination in ParameterGrid(param[model_name]):
                    model_instance = model.set_params(**param_combination)
                    y_pred = model_instance.fit_predict(X_train)
                    
                    unique_labels = np.unique(y_pred)
                    n_clusters = len(unique_labels[unique_labels != -1])  # Exclude noise points for DBSCAN
                    
                    logging.info(f"{model_name} with {param_combination}: {n_clusters} clusters found")
                    
                    if n_clusters <= 1:
                        logging.warning(f"{model_name} with {param_combination} did not find multiple clusters. Skipping.")
                        continue
                    
                    n_samples = X_train.shape[0] if not issparse(X_train) else X_train.shape[0]
                    if n_clusters >= n_samples:
                        logging.warning(f"{model_name} with {param_combination} created too many clusters. Skipping.")
                        continue
                    
                    X_train_dense = X_train.toarray() if issparse(X_train) else X_train
                    
                    score = silhouette_score(X_train_dense, y_pred)
                    logging.info(f"{model_name} with {param_combination}: Silhouette Score = {score}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = param_combination
                
                if best_params is None:
                    logging.warning(f"No valid clustering found for {model_name}. Skipping this model.")
                    continue
                
                best_model = model.set_params(**best_params)
                y_pred = best_model.fit_predict(X_train)
                model_report[model_name] = best_score
                logging.info(f"{model_name} - Best params: {best_params}, Silhouette Score: {best_score}")
        return model_report
    except Exception as e:
        logging.error(f"Error in evaluate_models: {str(e)}")
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)