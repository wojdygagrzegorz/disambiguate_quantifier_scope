import pickle
from abc import ABC, abstractmethod
from typing import Sequence, Union, Tuple

import pandas as pd


class Classifier(ABC):
    """
    Abstract base class for all classifiers.
    """
    def __init__(self, name, model=None):
        self.name = name
        self.model = model

    @abstractmethod
    def train(self, dataset: Union[Sequence, pd.DataFrame]) -> None:
        """
        Train the classifier on a list of sentences and labels.
        """
        pass

    @abstractmethod
    def predict(self, sentences: Union[Sequence, pd.DataFrame]) -> float:
        """
        Classify a list of sentences and return a list of labels.
        """
        pass

    @abstractmethod
    def transform_data(cls, data: Union[Sequence, pd.DataFrame]):
        """
        Transform data to the format required by the classifier.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the classifier to a file.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the classifier from a file.
        """
        pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name