import copy
import os
from pathlib import Path
from typing import Union

import fire
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.classifier import EnsembleVotingClassifier, GermEvalModel
from src.features import FeatureExtractor
from src.utils import Logger, multilabel_to_multiclass, read_config


def main(
    train_file: Union[str, Path],
    test_file: Union[str, Path],
    config_file: Union[str, Path],
    results_dir: Union[str, Path] = "./results",
    top_k: int = 25,
    show_progress_bar: bool = False,
) -> None:
    """
    Main function to train and evaluate our GermEval 2021 model, and generate predictions.

    This function orchestrates the process of feature extraction, model training, evaluation, and prediction.
    It performs the following steps:

    1. Reads configuration from a specified config file.
    2. Initializes a feature extractor based on the configuration.
    3. Extracts features from the training dataset and optionally saves them.
    4. Splits the training data into training and validation sets using StratifiedKFold.
    5. Trains the model on each fold and evaluates feature importance.
    6. Logs performance metrics for each fold.
    7. Extracts features from the test dataset and makes predictions using an ensemble of trained models.
    8. Saves predictions to CSV files for submission.

    Parameters
    ----------
    train_file : Union[str, Path]
        Path to the CSV file containing the training data.
    test_file : Union[str, Path]
        Path to the CSV file containing the test data.
    config_file : Union[str, Path]
        Path to the YAML configuration file specifying feature functions and model parameters.
    results_dir : Union[str, Path], optional
        Directory to save results, including feature files, model files, and logs. Defaults to "./results".
    top_k : int, optional
        Number of top features to display in feature importance plots. Defaults to 25.
    show_progress_bar : bool, optional
        Whether to display a progress bar during feature extraction. Defaults to False.

    Returns
    -------
    None
        This function does not return any value. It saves results and logs to specified files.

    """
    config = read_config(config_file)

    results_dir = os.path.join(results_dir, config["name"])
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    feature_extractor = FeatureExtractor(config["feature_funcs"])

    features_train, labels_train = feature_extractor.get_features(
        train_file,
        save_file=os.path.join(results_dir, "features_train.npz"),
        train=True,
        show_progress_bar=show_progress_bar,
    )

    splitter = StratifiedKFold(n_splits=config["num_splits"], shuffle=True, random_state=config["random_state"])
    multiclass_labels_train = multilabel_to_multiclass(labels_train)

    logger = Logger(os.path.join(results_dir, "training.log"))

    model_ensemble = []

    for fold, indices in enumerate(splitter.split(features_train, multiclass_labels_train)):
        train_idx, valid_idx = indices
        fold_features_train, fold_labels_train = features_train[train_idx], labels_train[train_idx]
        fold_features_valid, fold_labels_valid = features_train[valid_idx], labels_train[valid_idx]

        model = GermEvalModel(config["feature_funcs"])
        model.fit(
            fold_features_train,
            fold_labels_train,
            fold_features_valid,
            fold_labels_valid,
            num_trials=config["num_trials"],
            save_file=os.path.join(results_dir, "model_fold{}.pkl".format(fold)),
        )

        model_ensemble.append(copy.deepcopy(model.model))
        feature_importance = model.get_feature_importance(top_k=top_k)

        for task in feature_importance:
            importances, feature_names = feature_importance[task][0], feature_importance[task][1]

            importance_plot, ax = plt.subplots(nrows=1, ncols=1)
            ax.barh(range(len(feature_names)), importances, align="center")
            plt.title("Fold {}, Task: {} - Absolute Feature Importance (Top {})".format(fold, task, top_k))
            plt.xlabel("Absolute Feature Importance")
            plt.ylabel("Feature Name")
            plt.yticks(range(len(feature_names)), feature_names)
            plt.grid(True)

            plt.savefig(
                os.path.join(results_dir, "feature_importance_{}_fold{}.pdf".format(task.lower(), fold)),
                bbox_inches="tight",
            )
            plt.close()

        fold_predictions_valid = model.predict(fold_features_valid)
        logger.update(fold_labels_valid, fold_predictions_valid)

    logger.print_metrics()

    features_test, _ = feature_extractor.get_features(
        test_file, save_file=os.path.join(results_dir, "features_test.npz"), show_progress_bar=show_progress_bar
    )

    classifier = EnsembleVotingClassifier(model_ensemble)
    predictions_test = classifier.predict(features_test)

    submission_data_frame = pd.read_csv(test_file, header=0)
    submission_data_frame["Sub1_Toxic"] = predictions_test[:, 0]
    submission_data_frame["Sub2_Engaging"] = predictions_test[:, 1]
    submission_data_frame["Sub3_FactClaiming"] = predictions_test[:, 2]

    submission_data_frame.to_csv(os.path.join(results_dir, "submission_with_text.csv"), index=False, sep=",")
    submission_data_frame.drop(columns=["c_text"]).to_csv(
        os.path.join(results_dir, "submission.csv"), index=False, sep=","
    )


if __name__ == "__main__":
    fire.Fire(main)
