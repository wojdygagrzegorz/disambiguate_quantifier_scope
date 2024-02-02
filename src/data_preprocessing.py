import pandas as pd

from pydantic import BaseModel, Field

from typing import List

def transform2pandas(data):
    """Transforms data to pandas format.

    Args:
        data (list): List of lists, where each sublist is a list of
            strings, where the first element is the label and the
            rest are the features.

    Returns:
        pandas.DataFrame: DataFrame with the first column as the label
            and the rest as the features.

    """
    return pd.DataFrame(data, columns=['label'] + ['feature' + str(i) for i in range(len(data[0]) - 1)]

def transform2sklearn_format(data):
    """Transforms data to sklearn format.

    Args:
        data (list): List of lists, where each sublist is a list of
            strings, where the first element is the label and the
            rest are the features.

    Returns:
        list: List of tuples, where each tuple is a pair of a label
            and a list of features.

    """
    return [(row[0], row[1:]) for row in data]

def transform2transformer_format(data):
    """Transforms data to transformer format.

    Args:
        data (list): List of lists, where each sublist is a list of
            strings, where the first element is the label and the
            rest are the features.

    Returns:
        list: List of dicts, where each dict has a key 'text' and a
            key 'labels'.

    """
    return [{'text': row[1], 'labels': row[0]} for row in data]