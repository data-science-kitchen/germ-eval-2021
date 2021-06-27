from features import FeatureExtractor, NumCharacters, NumTokens
import fire
from model import GermEvalModel
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from typing import Optional, Union
from utils import multilabel_to_multiclass


def main(train_file: Union[str, Path],
         test_file: Union[str, Path],
         tmp_dir: Union[str, Path] = './tmp',
         num_splits: int = 5,
         num_trials: int = 100,
         show_progress_bar: bool = False,
         random_state: int = 42) -> None:
    feature_funcs = [NumCharacters(apply_log=True), NumTokens(apply_log=True)]
    feature_extractor = FeatureExtractor(feature_funcs)
    
    features_train, labels_train = feature_extractor.get_features(train_file,
                                                                  save_file=os.path.join(tmp_dir, 'features_train.npz'),
                                                                  show_progress_bar=show_progress_bar)

    splitter = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)

    for train_idx, valid_idx in splitter.split(features_train, multilabel_to_multiclass(labels_train)):
        fold_features_train, fold_labels_train = features_train[train_idx], labels_train[train_idx]

        model = GermEvalModel(feature_funcs)
        model.fit(fold_features_train, fold_labels_train)

        raise NotImplementedError
    
    features_test, _ = feature_extractor.get_features(test_file,
                                                      save_file=os.path.join(tmp_dir, 'features_test.npz'),
                                                      show_progress_bar=show_progress_bar)
    
    raise NotImplementedError


if __name__ == '__main__':
    fire.Fire(main)
