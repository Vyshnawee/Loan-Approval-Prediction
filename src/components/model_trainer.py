import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.util import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {

                "Logistic Regression": LogisticRegression(
                    max_iter=1000
                ),

                "Decision Tree": DecisionTreeClassifier(
                    random_state=42
                ),

                "Random Forest": RandomForestClassifier(
                    random_state=42
                ),

                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42
                ),

                "K-Neighbors": KNeighborsClassifier(),

                "XGBoost": XGBClassifier(
                    random_state=42,
                    eval_metric="logloss"
                ),

                "CatBoost": CatBoostClassifier(
                    verbose=False,
                    random_state=42
                ),

                "AdaBoost": AdaBoostClassifier(
                    random_state=42
                )
            }

            params = {

               "Logistic Regression": {
                    "C": [1],
                    "solver": ["liblinear"]
               },

               "Decision Tree": {
                    "criterion": ["gini"],
                    "max_depth": [5, 10]
               },

               "Random Forest": {
                    "n_estimators": [100],
                    "max_depth": [10]
               },

               "Gradient Boosting": {
                    "learning_rate": [0.1],
                    "n_estimators": [100],
                    "max_depth": [3]
               },

               "K-Neighbors": {
                    "n_neighbors": [5],
                    "weights": ["uniform"]
               },

               "XGBoost": {
                    "learning_rate": [0.1],
                    "n_estimators": [100],
                    "max_depth": [3]
               },

               "CatBoost": {
                    "depth": [6],
                    "learning_rate": [0.1],
                    "iterations": [100]
               },

               "AdaBoost": {
                    "learning_rate": [0.1],
                    "n_estimators": [100]
               }
          }

            model_report, best_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_score = max(
                model_report.values()
            )

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(
                    best_model_score
                )
            ]

            best_model = best_models[best_model_name]
            if best_model_score < 0.60:
                raise CustomException(
                    "No best model found"
                )

            logging.info(f"Best model: {best_model_name}")

            logging.info(f"Best model F1 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test,predicted)

            f1 = f1_score(y_test,predicted,average="weighted")

            precision = precision_score(y_test,predicted,average="weighted")

            recall = recall_score(y_test,predicted,average="weighted")

            return {
                "best_model": best_model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }


        except Exception as e:

            raise CustomException(e, sys)