import os
from pathlib import Path
from typing import Union

import pandas as pd
from flair.datasets import CSVClassificationCorpus


class GermEval2021(CSVClassificationCorpus):
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
