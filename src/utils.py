import os
import sys
import pickle
import dill
import  pandas as pd
import  numpy as np

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import  GridSearchCV

def save_pipeline_object(path, object):
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)

        # Saving the file as pickle file
        pickle.dump(object, open(path, 'wb'))

        # with open(path, 'wb') as file_obj:
        #     dill.dump(object, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def get_pickle_file(file_path):

    try:

        with open(file_path, 'rb') as file_obj:
            file = pickle.load((file_obj))
        return  file

    except Exception as e:
        raise  CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            hype_param = params[list(models.keys())[i]]
            clf = GridSearchCV(model, hype_param, cv=5)
            clf.fit(X_train, y_train)

            # Model training
            model.set_params(**clf.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred  = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score  = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_score
            return report

    except Exception as e:
        raise CustomException(e, sys)