from dataset import GermEval2021, GermEval2021Test
from features import *
import fire
from flair.embeddings import TransformerDocumentEmbeddings
import joblib
import optuna
import os
import pandas as pd
from pathlib import Path
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from typing import Optional, Tuple, Union

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD


class FeatureSplitter(BaseEstimator, TransformerMixin):
    def __init__(self,
                 feature_range: Tuple[int, int]):
        self.feature_range = feature_range

    def transform(self, X):
        return X[..., self.feature_range[0]:self.feature_range[1]]

    def fit(self, X, y):
        return self


def get_pipeline(feature_dim: int,
                 embedding_dim: int,
                 svd_num_components: Optional[int] = None,
                 svm_penalty: Optional[float] = 1.0):
    if svd_num_components is None:
        svd_num_components = embedding_dim

    pipeline = Pipeline([
        ('processed_features', FeatureUnion([
            ('numerical_processor', Pipeline([
                ('numerical_split', FeatureSplitter(feature_range=(0, feature_dim)))
            ])),
            ('embeddings_processor', Pipeline([
                ('embeddings_split', FeatureSplitter(feature_range=(feature_dim, feature_dim + embedding_dim))),
                ('embeddings_scaler', StandardScaler()),
                ('embeddings_svd', TruncatedSVD(n_components=svd_num_components))
             ])),
        ])),
        ('feature_scaler', StandardScaler()),
        ('classifier', MultiOutputClassifier(SVC(C=svm_penalty), n_jobs=-1))
    ])

    return pipeline


def objective_function(trial: optuna.Trial,
                       features_train: np.array,
                       labels_train: np.array,
                       features_dev: np.array,
                       labels_dev: np.array,
                       feature_dim: int,
                       embedding_dim: int) -> Tuple[float, float, float]:

    pipeline = get_pipeline(feature_dim,
                            embedding_dim,
                            trial.suggest_int('svd_num_components', 1, embedding_dim - 1),
                            trial.suggest_loguniform('svm_penalty', 0.1, 1e4))

    pipeline.fit(features_train, labels_train)

    predictions = pipeline.predict(features_dev)

    score_toxic = f1_score(labels_dev[:, 0], predictions[:, 0], average='macro')
    score_engaging = f1_score(labels_dev[:, 1], predictions[:, 1], average='macro')
    score_fact_claiming = f1_score(labels_dev[:, 2], predictions[:, 2], average='macro')

    return (score_toxic + score_engaging + score_fact_claiming) / 3.


def main(corpus_file: Union[str, Path],
         tmp_dir: Optional[Union[str, Path]] = None,
         num_trials: int = 100) -> None:
    if tmp_dir is not None and not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)

    results_list = []
    tasks = ['Toxic', 'Engaging', 'FactClaiming']

    document_embeddings = TransformerDocumentEmbeddings('bert-base-german-cased', fine_tune=False)

    features = [
        log_num_characters, log_average_word_length, log_word_length_std, positive_sentiment_logits,
        negative_sentiment_logits, neutral_sentiment_logits, spelling_mistake_log_odds
    ]

    feature_extractor = FeatureExtractor(features, document_embeddings=document_embeddings)

    with tqdm(total=4) as progress_bar:
        for fold_idx in range(4):
            corpus = GermEval2021(corpus_file, fold=fold_idx)

            if tmp_dir is not None:
                save_file = os.path.join(tmp_dir, 'features_fold{}.npz'.format(fold_idx))
                model_file = os.path.join(tmp_dir, 'model_fold{}.pkl'.format(fold_idx))
            else:
                save_file = None
                model_file = None

            data_train, data_dev = feature_extractor.compute_features(corpus, save_file=save_file)

            features_train, labels_train = data_train
            features_dev, labels_dev = data_dev

            if os.path.isfile(model_file):
                pipeline = joblib.load(model_file)
            else:
                study = optuna.create_study(directions=['maximize'])

                feature_dim, embedding_dim = len(features), document_embeddings.embedding_length

                study.optimize(
                    lambda x: objective_function(x, features_train, labels_train, features_dev, labels_dev, feature_dim,
                                                 embedding_dim),
                    n_trials=num_trials,
                    gc_after_trial=True
                )

                pipeline = get_pipeline(feature_dim,
                                        embedding_dim,
                                        study.best_params['svd_num_components'],
                                        study.best_params['svm_penalty'])

                pipeline.fit(features_train, labels_train)

                joblib.dump(pipeline, model_file)

            predicted_labels = pipeline.predict(features_dev)

            for task_idx, task in enumerate(tasks):
                results_list.append({
                    'accuracy': accuracy_score(labels_dev[:, task_idx], predicted_labels[:, task_idx]),
                    'precision': precision_score(labels_dev[:, task_idx], predicted_labels[:, task_idx]),
                    'recall': recall_score(labels_dev[:, task_idx], predicted_labels[:, task_idx]),
                    'f1': f1_score(labels_dev[:, task_idx], predicted_labels[:, task_idx], average='macro'),
                    'fold_idx': fold_idx,
                    'task': task
                })

            progress_bar.update()

    results = pd.DataFrame(results_list)

    for task in tasks:
        accuracy_mean = results[results['task'] == task].accuracy.mean()
        accuracy_std = results[results['task'] == task].accuracy.std()
        precision_mean = results[results['task'] == task].precision.mean()
        precision_std = results[results['task'] == task].precision.std()
        recall_mean = results[results['task'] == task].recall.mean()
        recall_std = results[results['task'] == task].recall.std()
        f1_score_mean = results[results['task'] == task].f1.mean()
        f1_score_std = results[results['task'] == task].f1.std()

        print('{:13s} === '
              'Accuracy: {:0.4f} +/- {:0.4f}, '
              'Precision: {:0.4f} +/- {:0.4f}, '
              'Recall: {:0.4f} +/- {:0.4f}, '
              'F1-Score: {:0.4f} +/- {:0.4f}'.format(task, accuracy_mean, accuracy_std, precision_mean, precision_std,
                                                     recall_mean, recall_std, f1_score_mean, f1_score_std))

    test_corpus = GermEval2021Test('./data/GermEval21_Toxic_TestData.csv')
    save_file = os.path.join(tmp_dir, 'features_test.npz')
    data_test, _ = feature_extractor.compute_features(test_corpus, save_file=save_file)


if __name__ == '__main__':
    fire.Fire(main)
