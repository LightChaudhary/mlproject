import os
import sys
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
    """Holds file paths for the data transformation stage artifacts."""

    # Serialized preprocessor output path; loaded during inference.
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    Handles preprocessing of train/test data using configured transformation logic.
    """

    def __init__(self):
        # Load transformation config (artifact paths, parameters)
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Build and return the preprocessing pipeline for student performance data.

        Constructs separate pipelines for numerical and categorical features, then combines them into a single ColumnTransformer.

        Returns:
            ColumnTransformer: Configured preprocessing pipeline.

        Raises:
            CustomException: If pipeline construction fails.
        """
        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            # Numerical: median imputation -> standard scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical: mode imputation -> one-hot encoding -> scaling (no centering)
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            logging.info(f"Preprocessor: {preprocessor}")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Load raw data, apply preprocessing, and persist the fitted transformer.

        Reads train/test CSVs, splits features from target, fits the preprocessor on training data, transforms both datasets, and saves the preprocessor artifact.

        Args:
            train_path: Path to the raw training CSV.
            test_path: Path to the raw testing CSV.
        
        Returns:
            Tuple of (train array, test array, preprocessor_file_path) where arrays have features and target concatenated as the last column.
        
        Raises:
            CustomException: If reading, transformation, or saving fails.
        """
        try:
            # Load raw datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info(f"Train DataFrame head: \n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame head: \n{test_df.head().to_string()}")

            # Build preprocessing pipeline
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Separate features and target
            target_column_name = "math score"
            numerical_columns = ["writing score", "reading score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Fit on train, transform both (prevents data leakage)
            logging.info("Applying preprocessing object on training and testing datasets")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatenate transformed features with target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Persist preprocessor for inference
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info(f"Saved preprocessing object.")

            logging.info(f"Train array shape: {train_arr.shape}")
            logging.info(f"Test array shape: {test_arr.shape}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)