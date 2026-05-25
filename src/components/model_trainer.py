import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


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
            logging.info("Splitting training and test input data")

            # Splitting train and test arrays
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info("Model training started")

            models = {

                "Linear Regression":
                LinearRegression(),

                "K-Neighbors Regressor":
                KNeighborsRegressor(),

                "Decision Tree":
                DecisionTreeRegressor(),

                "Random Forest":
                RandomForestRegressor(),

                "Gradient Boosting":
                GradientBoostingRegressor(),

                "XGBoost":
                XGBRegressor(),

                "CatBoost":
                CatBoostRegressor(verbose=False),

                "AdaBoost":
                AdaBoostRegressor()
            }

            # Evaluate all models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            print("\n================ Model Report ================\n")
            print(model_report)

            logging.info(f"Model Report : {model_report}")

            # Get best model score
            best_model_score = max(
                sorted(model_report.values())
            )

            # Get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Get best model object
            best_model = models[best_model_name]

            print(f"\nBest Model Found : {best_model_name}")
            print(f"Best Model Score : {best_model_score}")

            logging.info(
                f"Best Model Found : {best_model_name}"
            )

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Best model saved successfully")

            # Predictions using best model
            predicted = best_model.predict(X_test)

            # Final R2 Score
            r2_square = r2_score(y_test, predicted)

            print(f"\nFinal R2 Score : {r2_square}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)