from typing import Any, Dict

import importlib
import os
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.base import BaseEstimator

from parse import yamlobj


@yamlobj("!ClfWrapper")
class ClfWrapper(BaseEstimator):
    yaml_tag = u'!ClfWrapper'
    """Classifier wrapper for desired estimator."""
    def __init__(self, clf_module: str, clf_name: str, clf_params: Dict[str,
                                                                        Any]):
        """Initializes ClfWrapper.

        Inputs:
            clf_module: str
                Name of module from which to import classifier
            clf_name: str
                Name of classifier
            clf_params: dict
                Parameters to instantiate classifier with
        """
        super(ClfWrapper, self).__init__()
        lib = importlib.import_module(clf_module, clf_name)
        self.estimator = getattr(lib, clf_name)(**clf_params)
        self.name = clf_name

    def fit(self, x, y=None, **kwargs):
        """Fits estimator with provided data.

        Inputs:
            x: np.ndarray or pandas dataframe
                Training data
            y: np.ndarray or pandas dataframe
                Training labels
            kwargs: kwargs dict
                Any additional relevant kwargs for fitting data

        Returns:
            A ClfWrapper object with trained estimator.
        """

        self.estimator.fit(x, y)
        return self

    def predict(self, x, y=None):
        """Predicts labels for samples in x.

        Inputs:
            x: np.ndarray or pandas dataframe
                Prediction data
            y: np.ndarray or pandas dataframe
                Prediction labels

        Returns:
            Predicted class label for each sample.
        """

        return self.estimator.predict(x)

    def predict_proba(self, x):
        """Predicts probabilities.

        Inputs:
            x: np.ndarray or pandas dataframe
                Prediction data

        Returns:
            Array-like of shape (n_samples, n_classes) of probability
            of the sample for each class in the model.
        """

        return self.estimator.predict_proba(x)

    def transform(self, x):
        return self.predict_proba(x)

    def score(self, x, y) -> float:
        """Evaluates accuracy of estimator on data.

        Inputs:
            x: np.ndarray or pandas dataframe
                Evaluation data
            y: np.ndarray or pandas dataframe
                Evaluation labels

        Returns:
            Accuracy of estimator on data as float value.
        """

        return self.estimator.score(x, y)

    def get_metric(self, x, y, metric: str):
        """Selects metric on which to evaluate data.

        Inputs:
            x: np.ndarray or pandas dataframe
                Evaluation data
            y: np.ndarray or pandas dataframe
                Evaluation labels
            metric: str
                Metric to use to evaluate data

        Returns:
            Performance of estimator on provided metric. Output type
            is metric-specific, but currently outputs float.
        """

        if metric == 'score' or metric == 'accuracy':
            return self.score(x, y)
        elif metric == 'precision':
            y_pred = self.predict(x)
            from sklearn.metrics import precision_score
            return precision_score(y, y_pred)
        elif metric == 'recall':
            y_pred = self.predict(x)
            from sklearn.metrics import recall_score
            return recall_score(y, y_pred)
        else:
            print("Not supported.")

    def save_model(self, save_folder: str, save_file: str):
        """Saves model in desired folder and file.

        Inputs:
            save_folder: str
                Folder in which to save model
            save_file
                File in which to save model
        """

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        dump(self.estimator, os.path.join(save_folder, save_file + '.joblib'))

    def save_res(self, save_folder: str, save_file: str, metric: str, res):
        """Saves results in desired folder and file.

        Inputs:
            save_folder: str
                Folder in which to save results
            save_file: str
                File in which to save results
            metric: str
                Metric used to generate results
            res: float, integer, np.ndarray, or list
                Results to save
        """

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not isinstance(res, (list, np.ndarray)):
            res = [res]
        try:
            save_data = pd.DataFrame(res, columns=[metric])
            save_data.to_csv(
                os.path.join(save_folder, save_file + '_' + metric + '.csv'))
        except ValueError:
            print("Cannot convert data to dataframe.")


def train_model(clf,
                x_train,
                y_train,
                model_save_folder='scratch/',
                model_save_name='model/'):
    fitted_clf = clf.fit(x_train, y_train)
    clf.save_model(model_save_folder, model_save_name)
    return fitted_clf


def evaluate_model(clf,
                   x,
                   y,
                   metric,
                   res_save_folder='res/',
                   res_save_fname='test_res',
                   save_res=True):
    res = clf.get_metric(x, y, metric)
    if save_res:
        clf.save_res(res_save_folder, res_save_fname, metric, res)
    return res
