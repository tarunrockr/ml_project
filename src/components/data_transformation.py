import sys
import os

from dataclasses import  dataclass
from src.exception import CustomException
from src.logger import logging
from src import utils

import pandas as pd
import numpy  as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import  SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns   = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            # logging.info("Categorical columns: ",categorical_columns)
            # logging.info("Numerical columns: ", numerical_columns)
            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', num_pipeline, numerical_columns),
                    ('categorical_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return  preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_dataframe = pd.read_csv(train_path)
            test_dataframe  = pd.read_csv(test_path)
            logging.info("Train and test data import complete.")

            preprocessor_object = self.get_data_transformer_object()
            logging.info("Fetching preprocessor object")

            target_column_name = 'math_score'

            logging.info("Seperating input and output features from train and test dataframe")
            input_train_dataframe = train_dataframe.drop(columns=[target_column_name], axis=1)
            target_train_dataframe= train_dataframe[target_column_name]

            input_test_dataframe = test_dataframe.drop(columns=[target_column_name], axis=1)
            target_test_dataframe= test_dataframe[target_column_name]

            print("preprocessor_object",preprocessor_object)

            # Output of this transformed result will be a numpy array
            input_train_transformed_dataframe = preprocessor_object.fit_transform(input_train_dataframe)
            input_test_transformed_dataframe  = preprocessor_object.transform(input_test_dataframe)

            # print("test check 1: ",input_test_transformed_dataframe.shape)
            # n1 = np.array(target_test_dataframe).reshape((200,1))
            # print("train check 2: ", n1.shape)

            # Concatenating independent and dependent features in train dataframe
            # 1st way
            train_array = np.concatenate((input_train_transformed_dataframe, np.array(target_train_dataframe).reshape((800,1))), axis=1)
            # # 2nd way
            # train_array = np.c_[input_train_transformed_dataframe, np.array(target_train_dataframe)]

            # 1st way
            test_array = np.concatenate((input_test_transformed_dataframe, np.array(target_test_dataframe).reshape((200,1))), axis=1)
            # # 2nd way
            # test_array = np.c_[input_test_transformed_dataframe, np.array(target_test_dataframe)]

            logging.info("Saving the pipeline(preprocessor) as pickle file")
            utils.save_pipeline_object( path = self.data_transformation_config.preprocessor_obj_file_path, object=preprocessor_object )

            # Returning training array, test array and pipeline pickle file path
            return (train_array, test_array, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise  CustomException(e, sys)