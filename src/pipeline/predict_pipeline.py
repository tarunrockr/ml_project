import sys
import pandas as pd
import numpy  as np

from src.exception import CustomException
from src.utils import get_pickle_file

class PredictPipeline:
    def __init__(self, features_dataframe):
        self.features_dataframe = features_dataframe

    def predict(self):

        try:
            model_path        = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'

            # Applying feature engineering
            preprocessor     = get_pickle_file(file_path=preprocessor_path)
            transformed_data = preprocessor.transform(self.features_dataframe)

            # Predicting the output
            model           = get_pickle_file(file_path=model_path)
            prediction_data = model.predict(transformed_data)

            return  prediction_data
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:

    def __init__(self, gender: str, race_ethnicity: int, parental_level_of_education, lunch: int, test_preparation_course: int, reading_score: int, writing_score: int):
        self.gender                       = gender
        self.race_ethnicity               = race_ethnicity
        self.parental_level_of_education  = parental_level_of_education
        self.lunch                        = lunch
        self.test_preparation_course      = test_preparation_course
        self.reading_score                = reading_score
        self.writing_score                = writing_score

    def convert_data_in_dataframe(self):
        try:
            input_dict = {
                'gender':                      [self.gender],
                'race_ethnicity':              [self.race_ethnicity],
                'parental_level_of_education': [self.parental_level_of_education],
                'lunch':                       [self.lunch],
                'test_preparation_course':     [self.test_preparation_course],
                'reading_score':               [self.reading_score],
                'writing_score':               [self.writing_score]
            }
            return  pd.DataFrame(input_dict)

        except Exception as e:
            raise  CustomException(e, sys)