from datetime import datetime
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Union
import yaml


def read_config(config_file: Union[str, Path]) -> Dict:
    with open(config_file, "r") as file:
        config_data = yaml.load(file, Loader=yaml.SafeLoader)

    config_data["feature_funcs"] = []

    for feature in config_data["features"]:
        try:
            feature_func = getattr(__import__("preprocessing.features", fromlist=[feature]), feature)
            config_data["feature_funcs"].append(feature_func())
        except BaseException as err:
            print("Error when loading module {}: {}".format(feature, str(err)))

    return config_data


def multilabel_to_multiclass(labels: np.array) -> np.array:
    return LabelEncoder().fit_transform(["".join(str(label)) for label in labels])


class Logger:
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
