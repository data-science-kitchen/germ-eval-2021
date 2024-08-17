import os
from pathlib import Path
from typing import Union

import pandas as pd
from flair.datasets import CSVClassificationCorpus


class GermEval2021(CSVClassificationCorpus):
    """
    A dataset class for handling the GermEval 2021 dataset, extending the `CSVClassificationCorpus` from Flair.

    This class reads a CSV file containing text and labels for the GermEval 2021 shared task. It supports k-fold
    cross-validation by splitting the data into training and validation sets based on a specified fold number. The class
    automatically creates the necessary CSV files for training and validation within a specific directory structure.

    Parameters
    ----------
    base_path : Union[str, Path]
        The path to the CSV file containing the dataset. The file is expected to have a header row and columns separated
        by semicolons (`;`). The file should include a `fold` column to facilitate cross-validation splits.
    fold : int, optional
        The fold number to use for cross-validation. This determines which portion of the data will be used for
        validation. The default value is 0.
    **corpusargs : dict
        Additional arguments to pass to the `CSVClassificationCorpus` initializer.

    Attributes
    ----------
    fold_dir : Path
        The directory where the training and validation CSV files are stored for the specified fold.

    Methods
    -------
    None, but inherits methods from `CSVClassificationCorpus` for accessing the training, validation, and test datasets.
    """

    def __init__(self, base_path: Union[str, Path], fold: int = 0, **corpusargs):
        data_frame = pd.read_csv(base_path, header=0, sep=";")
        data_frame = data_frame.drop(columns=["comment_id"])

        fold_dir = Path(os.path.dirname(base_path), "fold_{}".format(fold))
        fold_dir.mkdir(parents=True, exist_ok=True)

        dev_fold_indices = list(data_frame[data_frame.fold == fold].index)

        training_data = (
            data_frame.drop(data_frame.index[dev_fold_indices]).reset_index(drop=True).drop(columns=["fold"])
        )
        training_data.to_csv(fold_dir / "train.csv", index=False)

        dev_data = data_frame.iloc[data_frame.index[dev_fold_indices]].reset_index(drop=True).drop(columns=["fold"])
        dev_data.to_csv(fold_dir / "dev.csv", index=False)

        super(GermEval2021, self).__init__(
            data_folder=fold_dir,
            column_name_map={0: "text", 1: "label_toxic", 2: "label_engaging", 3: "label_fact_claiming"},
            skip_header=True,
            train_file="train.csv",
            dev_file="dev.csv",
            test_file="dev.csv",  # Test file is not used during cross-validation
            **corpusargs,
        )


class GermEval2021Test(CSVClassificationCorpus):
    """
    A dataset class for handling the test set of the GermEval 2021 task, extending the `CSVClassificationCorpus` from Flair.

    This class reads a CSV file containing the test data for the GermEval 2021 shared task. The class prepares the data
    by saving it into a specific directory structure and initializing it as a corpus that can be used with the Flair
    framework.

    Parameters
    ----------
    test_file : Union[str, Path]
        The path to the CSV file containing the test dataset. The file is expected to have a header row and columns
        separated by commas (`,`).
    **corpusargs : dict
        Additional arguments to pass to the `CSVClassificationCorpus` initializer.

    Attributes
    ----------
    test_dir : Path
        The directory where the test CSV file is stored.

    Methods
    -------
    None, but inherits methods from `CSVClassificationCorpus` for accessing the test dataset.
    """

    def __init__(self, test_file: Union[str, Path], **corpusargs):
        data_frame = pd.read_csv(test_file, header=0, sep=",")

        test_dir = Path(os.path.dirname(test_file), "test")
        test_dir.mkdir(parents=True, exist_ok=True)

        data_frame.to_csv(test_dir / "test.csv", index=False)

        super(GermEval2021Test, self).__init__(
            data_folder=test_dir,
            column_name_map={0: "comment_id", 1: "text"},
            skip_header=True,
            train_file="test.csv",
            dev_file="test.csv",
            test_file="test.csv",
            **corpusargs,
        )
