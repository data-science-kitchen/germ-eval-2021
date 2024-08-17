from datetime import datetime
from pathlib import Path
from typing import Dict, Union

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


def read_config(config_file: Union[str, Path]) -> Dict:
    """
    Reads and parses a configuration file to generate a dictionary of configuration settings.

    This function loads a YAML configuration file, processes the list of feature names specified in the configuration,
    and attempts to dynamically import and instantiate feature functions. It adds these feature functions to the
    configuration dictionary under the key `"feature_funcs"`.

    Parameters
    ----------
    config_file : Union[str, Path]
        Path to the YAML configuration file. This file should contain a `"features"` key with a list of feature names
        that are to be dynamically imported and instantiated.

    Returns
    -------
    Dict
        A dictionary containing the configuration settings read from the YAML file. This dictionary will include:
        - All original keys from the YAML file.
        - An additional `"feature_funcs"` key, which contains a list of instantiated feature functions corresponding to
          the names specified in the `"features"` key of the YAML file.

    Raises
    ------
    FileNotFoundError
        If the specified configuration file does not exist.
    yaml.YAMLError
        If there is an error parsing the YAML file.
    ImportError
        If there is an error importing any of the specified feature modules.
    """
    with open(config_file, "r") as file:
        config_data = yaml.load(file, Loader=yaml.SafeLoader)

    config_data["feature_funcs"] = []

    for feature in config_data["features"]:
        try:
            feature_func = getattr(__import__("src.features", fromlist=[feature]), feature)
            config_data["feature_funcs"].append(feature_func())
        except BaseException as err:
            print("Error when loading module {}: {}".format(feature, str(err)))

    return config_data


def multilabel_to_multiclass(labels: np.array) -> np.array:
    """
    Convert multilabel binary labels into multiclass integer labels.

    This function takes an array of multilabel binary labels and converts it into a single array of integer labels.
    Each unique combination of binary labels is transformed into a unique integer value.

    Parameters
    ----------
    labels : np.array
        A 2D NumPy array where each row represents a set of binary labels in a multilabel classification. Each element
        in the row is expected to be binary (0 or 1).

    Returns
    -------
    np.array
        A 1D NumPy array where each element represents the integer encoding of the corresponding row in the input
        `labels` array. The integer values are obtained by encoding each unique combination of binary labels into a
        unique integer.

    Notes
    -----
    This function uses `LabelEncoder` from scikit-learn to perform the encoding. The binary labels are first converted
    to strings to ensure unique combinations are correctly encoded as distinct integers.
    """
    return LabelEncoder().fit_transform(["".join(str(label)) for label in labels])


class Logger:
    """
    A class for logging and calculating performance metrics for classification tasks.

    The `Logger` class manages the logging of performance metrics to a specified file. It supports multiple tasks and
    calculates common classification metrics such as accuracy, precision, recall, and F1-score.

    Attributes
    ----------
    TASK_NAMES : list of str
        A list of task names corresponding to the classification tasks. Each task name is used for labeling the metrics
        in the log file.
    log_file : Union[str, Path]
        The path to the file where the logs will be written.
    metrics_list : list of dict
        A list of dictionaries where each dictionary contains metrics for a specific task.

    Parameters
    ----------
    log_file : Union[str, Path]
        The path to the log file where metrics will be recorded.

    Methods
    -------
    update(labels: np.array, predictions: np.array) -> None
        Appends the computed metrics to the log file for the current timestamp.
    print_metrics()
        Prints the aggregated metrics (mean and standard deviation) for each task.
    _compute_metrics(labels: np.array, predictions: np.array) -> Dict
        Computes classification metrics for given labels and predictions.

    """

    TASK_NAMES = ["Toxic", "Engaging", "FactClaiming"]

    def __init__(self, log_file: Union[str, Path]) -> None:
        self.log_file = log_file
        self.metrics_list = []

        with open(self.log_file, "w") as file:
            file.write("Logging start: {}\n\n".format(datetime.now()))

    def update(self, labels: np.array, predictions: np.array) -> None:
        with open(self.log_file, "a") as file:
            file.write("{}\n".format(datetime.now()))

            for task_idx, task in enumerate(self.TASK_NAMES):
                metrics = self._compute_metrics(labels[:, task_idx], predictions[:, task_idx])
                metrics["task"] = task

                file.write(
                    "{:13s} === Accuracy: {:0.4f}, Precision: {:0.4f}, Recall: {:0.4f}, F1-Score: {:0.4f}\n".format(
                        task, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]
                    )
                )

                self.metrics_list.append(metrics)

            file.write("\n")

    def print_metrics(self):
        for task in self.TASK_NAMES:
            accuracy = np.array([x["accuracy"] for x in self.metrics_list if x["task"] == task])
            precision = np.array([x["precision"] for x in self.metrics_list if x["task"] == task])
            recall = np.array([x["recall"] for x in self.metrics_list if x["task"] == task])
            f1 = np.array([x["f1"] for x in self.metrics_list if x["task"] == task])

            print(
                "{:13s} === "
                "Accuracy: {:0.4f} +/- {:0.4f}, "
                "Precision: {:0.4f} +/- {:0.4f}, "
                "Recall: {:0.4f} +/- {:0.4f}, "
                "F1-Score: {:0.4f} +/- {:0.4f}".format(
                    task,
                    accuracy.mean(),
                    accuracy.std(),
                    precision.mean(),
                    precision.std(),
                    recall.mean(),
                    recall.std(),
                    f1.mean(),
                    f1.std(),
                )
            )

    @staticmethod
    def _compute_metrics(labels: np.array, predictions: np.array) -> Dict:
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average="macro"),
            "recall": recall_score(labels, predictions, average="macro"),
            "f1": f1_score(labels, predictions, average="macro"),
        }
