import os
import sys
import dill

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    param
):
    try:

        report = {}
        best_models = {}

        for model_name, model in models.items():
            para = param[model_name]

            # GridSearchCV
            gs = GridSearchCV(
                estimator=model,
                param_grid=para,
                cv=3,
                scoring="f1_weighted",
                n_jobs=-1
            )

            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_

            best_models[model_name] = best_model
            y_test_pred = best_model.predict(X_test)

            test_model_score = f1_score(
                y_test,
                y_test_pred,
                average="weighted"
            )

            report[model_name] = test_model_score

            print(
                f"{model_name}: "
                f"Best Parameters = {gs.best_params_}, "
                f"F1 = {test_model_score:.4f}"
            )

        return report, best_models

    except Exception as e:
        raise CustomException(e, sys)