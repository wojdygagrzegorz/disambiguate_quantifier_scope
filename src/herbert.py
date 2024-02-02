import copy
import logging

import numpy as np
import pandas as pd
import torch
from simpletransformers.classification import (ClassificationArgs,
                                               ClassificationModel)
from sklearn.metrics import accuracy_score

from classifier import Classifier
from src.pydantic_models import HerBERTConfiguration
from src.utils import configure_logging

HERBERT_CONFIG = 'uw_quantifiers/herbert_params.yaml'
logger = configure_logging(__name__)

class HerbertClassifier(Classifier):
    def __init__(self, name=None, configuration_path: str = HERBERT_CONFIG):
        if name is None:
            name = self.__class__.__name__
        super().__init__(name)
        self.conf = HerBERTConfiguration.load_from_file(configuration_path)
        self.conf.herbert_parameters['use_cuda'] = torch.cuda.is_available()

    def transform_data(self, data):
        colums_list = self.conf.data_parameters.input_columns + [self.conf.data_parameters.output_column]
        return data[colums_list]


    def train(self, data: pd.DataFrame):
        model_args = ClassificationArgs(**self.conf.model_parameters)
        model_params = self.conf.herbert_parameters | {'args': model_args}
        self.model = ClassificationModel(**model_params)
        train_data = self.transform_data(data)
        self.model.train_model(train_data)
        logger.info(f'Trained model {self.name}')

    def predict(self, data: pd.DataFrame):
        test_df = self.transform_data(data)
        _, model_outputs, _ = self.model.eval_model(test_df, acc=accuracy_score)
        return np.argmax(model_outputs, axis=1).tolist(), test_df[self.conf.data_parameters.output_column].values.tolist()


    def load(self, path):
        self.model = ClassificationModel(
            self.conf.herbert_parameters['model_type'], path, args=ClassificationArgs() # TODO add herbert to conf
        )

    def save(self, path=None):
        if path==None:
            path = self.conf.base_parameters.save_model_path
        self.model.save_model(path)
