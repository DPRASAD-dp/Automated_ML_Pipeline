import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    AdaBoostClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, f1_score, silhouette_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "best_model_info.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, problem_type):
        try:
            logging.info("Split training and test input data")
            
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            if problem_type == 'regression':
                models = {
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Linear Regression": LinearRegression(),
                    "AdaBoost Regressor": AdaBoostRegressor(),
                }
                params = {
                    "Decision Tree": {
                        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    },
                    "Random Forest": {
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "Linear Regression": {},
                    "AdaBoost Regressor": {
                        'learning_rate': [.1, .01, 0.5, .001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    }
                }
                metric = r2_score

            elif problem_type == 'classification':
                models = {
                    "Random Forest": RandomForestClassifier(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Logistic Regression": LogisticRegression(),
                    "AdaBoost Classifier": AdaBoostClassifier(),
                }
                params = {
                    "Decision Tree": {
                        'criterion': ['gini', 'entropy'],
                    },
                    "Random Forest": {
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    },
                    "Logistic Regression": {},
                    "AdaBoost Classifier": {
                        'learning_rate': [.1, .01, 0.5, .001],
                        'n_estimators': [8, 16, 32, 64, 128, 256]
                    }
                }
                metric = accuracy_score

            elif problem_type == 'clustering':
                models = {
                    "KMeans": KMeans(),
                    "DBSCAN": DBSCAN()
                }
                params = {
                    "KMeans": {'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10]},
                    "DBSCAN": {
                        'eps': [0.1, 0.5, 1.0, 1.5, 2.0],
                        'min_samples': [3, 5, 10, 20, 30]
                    }
                }
                metric = silhouette_score

            else:
                raise CustomException("Unsupported problem type")

            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                           models=models, param=params, metric=metric, problem_type=problem_type)

            if problem_type in ['regression', 'classification']:
                best_model_score = max(model_report.values(), key=lambda x: x[1])[1]  # Use test score for comparison
                best_model_name = max(model_report.items(), key=lambda x: x[1][1])[0]
            else:  # clustering
                best_model_score = max(model_report.values())
                best_model_name = max(model_report, key=model_report.get)

            return best_model_name, best_model_score, model_report

        except Exception as e:
            logging.error(f"Exception occurred in model training: {e}")
            raise CustomException(e, sys)