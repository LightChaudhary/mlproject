import os
import sys
import dill

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    """Serialize an object to disk with dill, creating parent dirs as needed."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate multiple regression models and return their r2 scores.
    """
    try:
        report = {}

        for model_name, model in models.items():
            param = params[model_name]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train) # Train model with GridSearchCV

            model.set_params(**gs.best_params_) # Set model parameters to best found by GridSearchCV
            model.fit(X_train, y_train) # Train model with best parameters

            y_train_pred = model.predict(X_train) # Predict on training data
            y_test_pred = model.predict(X_test) # Predict on test data  

            train_model_score = r2_score(y_train, y_train_pred) # Calculate r2 score for training data
            test_model_score = r2_score(y_test, y_test_pred) # Calculate r2 score for test data

            report[model_name] = {
                'train_r2': train_model_score,
                'test_r2': test_model_score
            } # Store r2 scores in report

        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)