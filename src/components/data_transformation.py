import os
import sys
import csv
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        '''
        This function is responsible for data transformation
        '''
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path, problem_type, target_column_name):
        try:
            train_df = pd.read_csv(train_path, 
                                   on_bad_lines='warn',    # Warn about skipped lines
                                   quoting=csv.QUOTE_MINIMAL,  # Only quote fields which contain special characters
                                   escapechar='\\')        # Use backslash as escape character
            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Train data columns: {train_df.columns.tolist()}")

            test_df = pd.read_csv(test_path, 
                                  on_bad_lines='warn',    # Warn about skipped lines
                                  quoting=csv.QUOTE_MINIMAL,  # Only quote fields which contain special characters
                                  escapechar='\\')        # Use backslash as escape character
            logging.info(f"Test data shape: {test_df.shape}")
            logging.info(f"Test data columns: {test_df.columns.tolist()}")

            logging.info("Read train and test data completed")

            # Automatically identify numerical and categorical columns
            numerical_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

            if target_column_name:
                if target_column_name in numerical_columns:
                    numerical_columns.remove(target_column_name)
                if target_column_name in categorical_columns:
                    categorical_columns.remove(target_column_name)

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(numerical_columns, categorical_columns)

            if problem_type in ['regression', 'classification']:
                if target_column_name is None:
                    raise CustomException("Target column name must be provided for regression and classification problems.")

                input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
                target_feature_train_df = train_df[target_column_name]

                input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
                target_feature_test_df = test_df[target_column_name]

                logging.info("Applying preprocessing object on training and testing dataframes.")

                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]
                test_arr = np.c_[
                    input_feature_test_arr, np.array(target_feature_test_df)
                ]

            elif problem_type == 'clustering':
                input_feature_train_df = train_df
                input_feature_test_df = test_df

                logging.info("Applying preprocessing object on training and testing dataframes for clustering.")

                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

                train_arr = input_feature_train_arr
                test_arr = input_feature_test_arr

            else:
                raise CustomException("Unsupported problem type")

            logging.info("Saved preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
