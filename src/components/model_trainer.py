import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        models,
        param
    ):

        try:

            report = {}
            best_params_report = {}
            best_models = {}

            for model_name, model in models.items():

                print(f"\n{'='*50}")
                print(f"Running Model: {model_name}")
                print(f"{'='*50}")

                logging.info(f"Running Model: {model_name}")

                params = param[model_name]

                gs = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=3,
                    error_score='raise'
                )

                # Grid Search Fit
                gs.fit(X_train, y_train)

                print(f"GridSearch Completed for {model_name}")

                logging.info(
                    f"GridSearch Completed for {model_name}"
                )

                # Best parameters
                best_params = gs.best_params_

                print(f"Best Params: {best_params}")

                logging.info(f"Best Params: {best_params}")

                # Set best params
                model.set_params(**best_params)

                # Train model with best params
                model.fit(X_train, y_train)

                # Save tuned model
                best_models[model_name] = model

                # Predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Scores
                train_model_score = r2_score(
                    y_train,
                    y_train_pred
                )

                test_model_score = r2_score(
                    y_test,
                    y_test_pred
                )

                # Store scores
                report[model_name] = test_model_score

                # Store best params
                best_params_report[model_name] = best_params

                print(f"Train R2 Score: {train_model_score}")
                print(f"Test R2 Score: {test_model_score}")

                logging.info(
                    f"{model_name} Train Score: {train_model_score}"
                )

                logging.info(
                    f"{model_name} Test Score: {test_model_score}"
                )

            return (
                report,
                best_params_report,
                best_models
            )

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            print("\nStarting Model Training Pipeline...\n")

            logging.info(
                "Splitting Training and Testing Input Data"
            )

            # Split train and test arrays
            X_train, y_train, X_test, y_test = (

                train_array[:, :-1],
                train_array[:, -1],

                test_array[:, :-1],
                test_array[:, -1]
            )

            # Models
            models = {

                "Random Forest": RandomForestRegressor(),

                "Decision Tree": DecisionTreeRegressor(),

                "Gradient Boosting": GradientBoostingRegressor(),

                "Linear Regression": LinearRegression(),

                "XGBRegressor": XGBRegressor(),

                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False
                ),

                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Hyperparameters
            params = {

                "Decision Tree": {

                    'criterion': [
                        'squared_error',
                        'friedman_mse'
                    ],

                    'splitter': [
                        'best',
                        'random'
                    ],

                    'max_depth': [
                        None,
                        5,
                        10,
                        20
                    ],
                },

                "Random Forest": {

                    'n_estimators': [
                        50,
                        100
                    ],

                    'max_depth': [
                        None,
                        10,
                        20
                    ],

                    'min_samples_split': [
                        2,
                        5
                    ],
                },

                "Gradient Boosting": {

                    'learning_rate': [
                        0.01,
                        0.1
                    ],

                    'n_estimators': [
                        100,
                        200
                    ],

                    'subsample': [
                        0.8,
                        1.0
                    ],
                },

                "Linear Regression": {

                    'fit_intercept': [
                        True,
                        False
                    ]
                },

                "XGBRegressor": {

                    'learning_rate': [
                        0.01,
                        0.1
                    ],

                    'n_estimators': [
                        100,
                        200
                    ],

                    'max_depth': [
                        3,
                        5,
                        7
                    ],
                },

                "CatBoosting Regressor": {

                    'depth': [
                        6,
                        8,
                        10
                    ],

                    'learning_rate': [
                        0.01,
                        0.05,
                        0.1
                    ],

                    'iterations': [
                        100,
                        200
                    ],
                },

                "AdaBoost Regressor": {

                    'learning_rate': [
                        0.01,
                        0.1,
                        1
                    ],

                    'n_estimators': [
                        50,
                        100,
                        200
                    ],
                }
            }

            # Evaluate models
            (
                model_report,
                best_params_report,
                best_models

            ) = self.evaluate_models(

                X_train=X_train,
                y_train=y_train,

                X_test=X_test,
                y_test=y_test,

                models=models,
                param=params
            )

            # Best Model Score
            best_model_score = max(
                sorted(model_report.values())
            )

            # Best Model Name
            best_model_name = list(
                model_report.keys()
            )[
                list(model_report.values()).index(
                    best_model_score
                )
            ]

            # Best Tuned Model
            best_model = best_models[best_model_name]

            # Final Output
            print("\n")
            print("=" * 60)
            print("FINAL MODEL REPORT")
            print("=" * 60)

            for model_name, score in model_report.items():

                print(
                    f"{model_name} --> Test R2 Score: {score}"
                )

            print("\n")
            print("=" * 60)
            print(f"BEST MODEL: {best_model_name}")
            print(f"BEST SCORE: {best_model_score}")
            print(
                f"BEST PARAMETERS: "
                f"{best_params_report[best_model_name]}"
            )
            print("=" * 60)

            logging.info(
                f"Best Model Found: {best_model_name}"
            )

            logging.info(
                f"Best Score: {best_model_score}"
            )

            logging.info(
                f"Best Params: "
                f"{best_params_report[best_model_name]}"
            )

            # Minimum threshold check
            if best_model_score < 0.6:

                raise CustomException(
                    "No Best Model Found",
                    sys
                )

            # Save Best Tuned Model
            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model
            )

            logging.info(
                "Best Model Saved Successfully"
            )

            # Final prediction
            predicted = best_model.predict(X_test)

            r2_square = r2_score(
                y_test,
                predicted
            )

            print(f"\nFinal R2 Score: {r2_square}")

            return (

                r2_square,

                best_model_name,

                best_params_report[best_model_name]
            )

        except Exception as e:

            print(e)

            logging.info(f"Exception Occurred: {e}")

            raise CustomException(e, sys)