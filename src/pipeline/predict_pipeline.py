import os
import sys
import dill

import pandas as pd

from src.exception import CustomException



# LOAD OBJECT FUNCTION

def load_object(file_path):

    try:

        with open(file_path, "rb") as file_obj:

            return dill.load(file_obj)

    except Exception as e:

        raise CustomException(e, sys)


# PREDICTION PIPELINE

class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:

            # MODEL PATH
            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            # PREPROCESSOR PATH
            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            # LOAD MODEL
            model = load_object(model_path)

            # LOAD PREPROCESSOR
            preprocessor = load_object(
                preprocessor_path
            )

            # TRANSFORM FEATURES
            data_scaled = preprocessor.transform(
                features
            )

            # PREDICT
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:

            raise CustomException(e, sys)


# CUSTOM INPUT DATA CLASS

class CustomData:

    def __init__(

        self,

        gender: str,

        race_ethnicity: str,

        parental_level_of_education: str,

        lunch: str,

        test_preparation_course: str,

        reading_score: int,

        writing_score: int

    ):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = (
            parental_level_of_education
        )

        self.lunch = lunch

        self.test_preparation_course = (
            test_preparation_course
        )

        self.reading_score = reading_score

        self.writing_score = writing_score

    # CONVERT INPUT TO DATAFRAME

    def get_data_as_data_frame(self):

        try:

            custom_data_input_dict = {

                "gender": [self.gender],

                "race_ethnicity": [
                    self.race_ethnicity
                ],

                "parental_level_of_education": [
                    self.parental_level_of_education
                ],

                "lunch": [
                    self.lunch
                ],

                "test_preparation_course": [
                    self.test_preparation_course
                ],

                "reading_score": [
                    self.reading_score
                ],

                "writing_score": [
                    self.writing_score
                ]

            }

            return pd.DataFrame(
                custom_data_input_dict
            )

        except Exception as e:

            raise CustomException(e, sys)