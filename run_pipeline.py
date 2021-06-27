import copy
from features import AverageTokenLength, DocumentEmbeddingsBERT, FeatureExtractor, NumCharacters, NumTokens, \
    SentimentBERT, SpellingMistakes, TokenLengthStandardDeviation
import fire
from model import EnsembleVotingClassifier, GermEvalModel
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from typing import Optional, Union
from utils import Logger, multilabel_to_multiclass


def main(train_file: Union[str, Path],
         test_file: Union[str, Path],
         tmp_dir: Union[str, Path] = './tmp',
         num_splits: int = 5,
         num_trials: int = 100,
         show_progress_bar: bool = False,
         random_state: int = 42) -> None:
    feature_funcs = [NumCharacters(apply_log=True), NumTokens(apply_log=True), AverageTokenLength(apply_log=True),
                     SpellingMistakes(), TokenLengthStandardDeviation(), SentimentBERT(), DocumentEmbeddingsBERT()]
    feature_extractor = FeatureExtractor(feature_funcs)
    
    features_train, labels_train = feature_extractor.get_features(train_file,
                                                                  save_file=os.path.join(tmp_dir, 'features_train.npz'),
                                                                  show_progress_bar=show_progress_bar)

    splitter = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    multiclass_labels_train = multilabel_to_multiclass(labels_train)

    logger = Logger(os.path.join(tmp_dir, 'training.log'))
    
    model_ensemble = []

    for fold, indices in enumerate(splitter.split(features_train, multiclass_labels_train)):
        train_idx, valid_idx = indices
        fold_features_train, fold_labels_train = features_train[train_idx], labels_train[train_idx]
        fold_features_valid, fold_labels_valid = features_train[valid_idx], labels_train[valid_idx]

        model = GermEvalModel(feature_funcs)        
        model.fit(fold_features_train, fold_labels_train, fold_features_valid, fold_labels_valid,
                  num_trials=num_trials, save_file=os.path.join(tmp_dir, 'model_fold{}.pkl'.format(fold)))

        model_ensemble.append(copy.deepcopy(model.model))

        fold_predictions_valid = model.predict(fold_features_valid)
        logger.update(fold_labels_valid, fold_predictions_valid)

    logger.print_metrics()

    features_test, _ = feature_extractor.get_features(test_file,
                                                      save_file=os.path.join(tmp_dir, 'features_test.npz'),
                                                      show_progress_bar=show_progress_bar)

    classifier = EnsembleVotingClassifier(model_ensemble)
    predictions_test = classifier.predict(features_test)

    submission_data_frame = pd.read_csv(test_file, header=0)
    submission_data_frame['Sub1_Toxic'] = predictions_test[:, 0]
    submission_data_frame['Sub2_Engaging'] = predictions_test[:, 1]
    submission_data_frame['Sub3_FactClaiming'] = predictions_test[:, 2]

    submission_data_frame.to_csv(os.path.join(tmp_dir, 'submission.csv'), index=False, sep=',')


if __name__ == '__main__':
    fire.Fire(main)
