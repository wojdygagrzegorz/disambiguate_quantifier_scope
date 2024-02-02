import functools
import operator
from typing import Sequence, List, Dict

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from src.utils import configure_logging, obtain_module_name

logger = configure_logging(obtain_module_name(__name__))

class ClassifierResults:
    def __init__(self, name) -> None:
        self.classifier_name = name
        self.predicted_labels: Sequence[int] = list()
        self.true_labels: Sequence[int] = list()
        self.accuracies: Sequence[float] = list()

    def add_results(self, predicted_label: Sequence[int], true_label: Sequence[int]):
        self.predicted_labels.append(predicted_label)
        self.true_labels.append(true_label)
        self.accuracies.append(accuracy_score(true_label, predicted_label))

    def calculate_general_metrics(self) -> float:
        self.predicted = functools.reduce(operator.iconcat, self.predicted_labels, [])
        self.true = functools.reduce(operator.iconcat, self.true_labels, [])
        self.f1 = f1_score(self.true, self.predicted, average='macro')
        self.recall = recall_score(self.true, self.predicted, average='macro')
        self.precision = precision_score(self.true, self.predicted, average='macro')
        self.accuracy = accuracy_score(self.true, self.predicted)

    def __str__(self):
        return f'Classifier {self.classifier_name} achieved {self.accuracy:.4f} accuracy, {self.precision:.4f} precision, {self.recall:.4f} recall {self.f1:.4f} F1'


class StatisticClass:
    def __init__(self, classifiers: Sequence[str], lenght: int):
        self.classifiers = classifiers
        self.results = dict()
        self.length = lenght
        for classifier in classifiers:
            self.results[classifier] = ClassifierResults(name=classifier)

    def add_results(self, classifier_name: str, predicted_labels: Sequence[int], true_labels: Sequence[int]):
        self.results[classifier_name].add_results(predicted_labels, true_labels)
        logger.info(f'Accuracy score for {classifier_name} is {accuracy_score(true_labels, predicted_labels):.4f}')

    def assert_all_true_label(self):
        first_classifier = self.classifiers[0]
        for classifier in self.classifiers:
            try:
                assert self.results[first_classifier].true_labels == self.results[classifier].true_labels
            except AssertionError as ae:
                logger.error(f'{first_classifier} and {classifier} has different true labels')
                raise AssertionError(ae)
        logger.info(f'Verified that all classfiers result had same true labels.')

    def print_general_statistic(self):
        self.assert_all_true_label()
        for classifier in self.classifiers:
            self.results[classifier].calculate_general_metrics()
            logger.info(str(self.results[classifier]))
        return self.results

    @staticmethod
    def calculate_stats_for_multiple_runs(results: List[Dict[str, ClassifierResults]]):
        logger.info(f'Calculating stats for {len(results)} runs')
        classifier_names = results[0].keys()
        for classifier_name in classifier_names:
            accuracies = [result[classifier_name].accuracy for result in results]
            precisions = [result[classifier_name].precision for result in results]
            recalls = [result[classifier_name].recall for result in results]
            f1s = [result[classifier_name].f1 for result in results]
            logger.info(f'Classifier {classifier_name}')
            logger.info(f'Average accuracy is {sum(accuracies)/len(accuracies):.4f}')
            logger.info(f'Average precision is {sum(precisions)/len(precisions):.4f}')
            logger.info(f'Average recall is {sum(recalls)/len(recalls):.4f}')
            logger.info(f'Average F1 is {sum(f1s)/len(f1s):.4f}')

