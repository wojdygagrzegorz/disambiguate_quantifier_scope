import copy
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import configure_logging, obtain_module_name

logger = configure_logging(obtain_module_name(__name__))


class Sampler:
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

    def run_balanced_dataset(self, data: pd.DataFrame, output_column: str, number_of_last_rows: int = 232):
        self.data = copy.copy(data)
        self.label_column = output_column
        logger.info(f'Running balanced dataset')
        train, test = self.data[:-number_of_last_rows], self.data[-number_of_last_rows:]
        yield train, test