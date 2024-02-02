import logging
import pickle
from typing import List

import fire
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.classifier import Classifier
from src.pydantic_models import RegressionConfiguarion
from src.utils import configure_logging, obtain_module_name

REGRESSION_CONFIG = 'uw_quantifiers/regression_params.yaml'
logger = configure_logging(obtain_module_name(__name__))


class RegressionClassifier(Classifier):
    def __init__(self, name=None, configuration_path: str = REGRESSION_CONFIG):
        if name is None:
            name = self.__class__.__name__
        super().__init__(name)
        self.conf = RegressionConfiguarion.load_from_file(configuration_path)

    def transform_data(self, data: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        return data[self.conf.data_parameters.input_columns], data[self.conf.data_parameters.output_column]

    def train(self, dataset: pd.DataFrame):
        self.model = Pipeline(steps=[("encoder", OneHotEncoder(**self.conf.encoder_parameters)),
                                     ("classifier", LogisticRegression(**self.conf.classifier_parameters))])
        X_train, y_train = self.transform_data(data=dataset)
        self.model.fit(X_train, y_train)
        logger.info(f"Trained model: {self.name}")

    def predict(self, dataset: pd.DataFrame) -> (List[int], List[int]):
        X_test, y_test = self.transform_data(dataset)
        predictions = self.model.predict(X_test)
        return predictions.tolist(), y_test.to_list()

    def save(self, path=None):
        """
        Save the classifier to a file.
        """
        pickle.dump(self.model, open(path, 'wb'))

    def load(self, path: str):
        """
        Load the classifier from a file.
        """
        self.model = pickle.load(open(path, 'rb'))


if __name__ == '__main__':
    fire.Fire(RegressionClassifier)
