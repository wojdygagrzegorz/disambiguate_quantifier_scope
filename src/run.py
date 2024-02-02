import fire
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from src.cross_validation import CrossValidation
from src.herbert import HerbertClassifier
from src.regression import RegressionClassifier
from src.utils import configure_logging, obtain_module_name

from sampler import Sampler
from statistic_class import StatisticClass

logger = configure_logging(obtain_module_name(__name__))
DEFAULT_RANDOME_STATE = 42

def run_cross_validation(data_file = None, random_state=DEFAULT_RANDOME_STATE):
    df = pd.read_csv(data_file)
    logger.info(f'Loaded {len(df)} rows data from {data_file}')

    cros_validator = CrossValidation(random_state=random_state)
    classifier_list = [RegressionClassifier(name="RegressionClassifier"),
                       HerbertClassifier(name="herbert_base", configuration_path='uw_quantifiers/herbert_params.yaml'),
                       HerbertClassifier(name="herbert_large", configuration_path='uw_quantifiers/herbert_large_params.yaml')]
    statistic_class = StatisticClass([classifier.name for classifier in classifier_list])
    output_column = classifier_list[0].conf.data_parameters.output_column

    for train, test in cros_validator.iterate(df, output_column):
        for classifier in classifier_list:
            classifier.train(train)
            predicted_labels, true_labels = classifier.predict(test)
            statistic_class.add_results(classifier.name, predicted_labels, true_labels)

    results = statistic_class.print_general_statistic()
    return results

def run_resample(data_file = None, random_state=DEFAULT_RANDOME_STATE):
    df = pd.read_csv(data_file)
    logger.info(f'Loaded {len(df)} rows data from {data_file}')

    sampler = Sampler(n_runs=10, random_state=random_state)
    classifier_list = [RegressionClassifier(name="RegressionClassifier"),
                       HerbertClassifier(name="herbert_base", configuration_path='uw_quantifiers/herbert_params.yaml')]
                       #HerbertClassifier(name="herbert_large", configuration_path='uw_quantifiers/herbert_large_params.yaml')]
    statistic_class = StatisticClass([classifier.name for classifier in classifier_list], lenght=len(df))
    output_column = classifier_list[0].conf.data_parameters.output_column

    for train, test in sampler.iterate(df, output_column):
        for classifier in classifier_list:
            classifier.train(train)
            predicted_labels, true_labels = classifier.predict(test)
            statistic_class.add_results(classifier.name, predicted_labels, true_labels)

    results = statistic_class.print_general_statistic()
    return results

def main(data_file = None, n_runs = 1):
    run_results = list()
    for i in range(n_runs):
        logger.info(f'Running {i+1} run')
        result = run_resample(data_file=data_file, random_state=DEFAULT_RANDOME_STATE+i)
        run_results.append(result)

    StatisticClass.calculate_stats_for_multiple_runs(run_results)

if __name__ == '__main__':
    fire.Fire(main)
