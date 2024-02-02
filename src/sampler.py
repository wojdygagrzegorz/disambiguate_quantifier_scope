import pandas as pd

import copy
import logging
from src.utils import configure_logging, obtain_module_name
import numpy as np
from sklearn.model_selection import train_test_split

logger = configure_logging(obtain_module_name(__name__))


class Sampler:   # TODO: add configuration
    def __init__(self, n_runs=5, random_state=42):
        self.n_run = n_runs
        self.random_state = random_state

    def iterate(self, data: pd.DataFrame, output_column: str):
        self.data = copy.copy(data.sample(frac=1).reset_index(drop=True))
        self.label_column = output_column
        rng = np.random.RandomState(self.random_state)
        for i in range(self.n_run):
            randint = rng.randint(low=0, high=32767)
            logger.info(f'Iteration {i+1} of {self.n_run} -> random state {randint}')
            train, test = train_test_split(self.data, test_size=0.10, random_state=randint)
            yield train, test