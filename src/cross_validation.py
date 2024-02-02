import copy
import logging
from typing import List

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.classifier import Classifier
from src.utils import configure_logging, obtain_module_name

logger = configure_logging(obtain_module_name(__name__))

DEFAULT_RANDOM_STATE = 42
DEFAULT_N_SPLIT = 5

class CrossValidation:
    def __init__(self, n_split=DEFAULT_N_SPLIT, random_state=DEFAULT_RANDOM_STATE):
        self.n_split = n_split
        self.random_state = random_state

    def iterate(self, data: pd.DataFrame, output_column: str):
        self.data = copy.copy(data.sample(frac=1, random_state=self.random_state).reset_index(drop=True))
        self.label_column = output_column
        logger.info(f'Loaded {len(self.data)} rows data to cross validation and shuffled.')
        self.skf = StratifiedKFold(n_splits=self.n_split, shuffle=True, random_state=42)
        logger.info(f'Prepared {self.n_split} stratified k-fold cross validation.')
        for i, (train_index, test_index) in enumerate(self.skf.split(self.data, self.data[self.label_column])):
            logger.info(f'Iteration {i+1} of {self.n_split}')
            train = self.data.iloc[train_index]
            test = self.data.iloc[test_index]
            yield train, test
