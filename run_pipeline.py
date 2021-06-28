import copy
from features import FeatureExtractor
import fire
from model import EnsembleVotingClassifier, GermEvalModel
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from typing import Union
from utils import Logger, multilabel_to_multiclass, read_config


def main(train_file: Union[str, Path],
         test_file: Union[str, Path],
         config_file: Union[str, Path],
         tmp_dir: Union[str, Path] = './tmp',
         show_progress_bar: bool = False) -> None:
    config = read_config(config_file)

    tmp_dir = os.path.join(tmp_dir, config['name'])
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)
    
    feature_extractor = FeatureExtractor(config['feature_funcs'])
    
    features_train, labels_train = feature_extractor.get_features(train_file,
                                                                  save_file=os.path.join(tmp_dir, 'features_train.npz'),
                                                                  show_progress_bar=show_progress_bar)

    splitter = StratifiedKFold(n_splits=config['num_splits'], shuffle=True, random_state=config['random_state'])
    multiclass_labels_train = multilabel_to_multiclass(labels_train)

    logger = Logger(os.path.join(tmp_dir, 'training.log'))
    
    model_ensemble = []

    for fold, indices in enumerate(splitter.split(features_train, multiclass_labels_train)):
        train_idx, valid_idx = indices
        fold_features_train, fold_labels_train = features_train[train_idx], labels_train[train_idx]
        fold_features_valid, fold_labels_valid = features_train[valid_idx], labels_train[valid_idx]

        model = GermEvalModel(config['feature_funcs'])
        model.fit(fold_features_train, fold_labels_train, fold_features_valid, fold_labels_valid,
                  num_trials=config['num_trials'], save_file=os.path.join(tmp_dir, 'model_fold{}.pkl'.format(fold)))

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
