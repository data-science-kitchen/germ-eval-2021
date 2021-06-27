from features import Feature
import joblib
import numpy as np
import optuna
import os
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from typing import List, Optional, Tuple, Union


class FeatureSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, mask: np.array) -> None:
        self.mask = mask

    def transform(self, X):
        return X[..., self.mask]

    def fit(self, X, y=None):
        return self


class EnsembleVotingClassifier(ClassifierMixin):
    def __init__(self, base_classifiers: List[ClassifierMixin]) -> None:
        self.base_classifiers = base_classifiers

    def predict(self, features: np.array) -> np.array:
        num_samples, num_classifiers = features.shape[0], len(self.base_classifiers)
        predictions = np.zeros((num_samples, num_classifiers, 3))

        for idx, classifier in enumerate(self.base_classifiers):
            predictions[:, idx, :] = classifier.predict(features)

        return (predictions.sum(axis=1) > num_classifiers / 2).astype(np.int64)


class GermEvalModel(ClassifierMixin):
    def __init__(self,
                 feature_funcs: List[Feature]) -> None:
        self.feature_funcs = feature_funcs
        self.model = None

        feature_types = self._get_feature_type()
        self.num_numerical_features = np.asarray([x == 'numerical' for x in feature_types], dtype=bool).sum()
        self.num_embedding_features = np.asarray([x == 'embedding' for x in feature_types], dtype=bool).sum()

    def fit(self,
            features_train: np.array,
            labels_train: np.array,
            features_valid: Optional[np.array] = None,
            labels_valid: Optional[np.array] = None,
            num_trials: int = 100,
            save_file: Optional[Union[str, Path]] = None) -> None:
        if save_file is not None and os.path.isfile(save_file):
            model = joblib.load(save_file)
        else:
            study = optuna.create_study(directions=['maximize'])
            study.optimize(lambda x: self._tuning_objective(x, features_train, labels_train, features_valid, labels_valid),
                           n_trials=num_trials,
                           gc_after_trial=True)

            model = self._get_model(
                svd_num_components=study.best_params['svd_num_components'], svm_penalty=study.best_params['svm_penalty']
            )
            model.fit(features_train, labels_train)

            joblib.dump(model, save_file)

        self.model = model

    def predict(self, features: np.array) -> np.array:
        if self.model is not None:
            return self.model.predict(features)
        else:
            raise ValueError('No trained model found. Please run .fit() first.')

    def _get_feature_type(self):
        feature_type = []

        for feature_func in self.feature_funcs:
            feature_type += feature_func.dim * [feature_func.type]

        return feature_type

    def _get_model(self,
                   svd_num_components: Optional[int] = None,
                   svm_penalty: Optional[float] = 1.0):
        feature_types = self._get_feature_type()
        numerical_features = np.asarray([x == 'numerical' for x in feature_types], dtype=bool)
        embedding_features = np.asarray([x == 'embedding' for x in feature_types], dtype=bool)

        if svd_num_components is None:
            svd_num_components = embedding_features.sum()

        feature_pipeline = []

        if numerical_features.sum() > 0:
            feature_pipeline += [('numerical_split', FeatureSplitter(numerical_features))]

        if embedding_features.sum() > 0:
            feature_pipeline += [('embedding_pipeline', Pipeline([
                ('embedding_split', FeatureSplitter(embedding_features)),
                ('embedding_scaler', StandardScaler()),
                ('embeddings_svd', TruncatedSVD(n_components=svd_num_components))
            ]))]

        model = Pipeline([
            ('features', FeatureUnion(feature_pipeline)),
            ('feature_scaler', StandardScaler()),
            ('classifier', MultiOutputClassifier(SVC(C=svm_penalty), n_jobs=-1))
        ])

        return model

    def _tuning_objective(self,
                          trial: optuna.Trial,
                          features_train: np.array,
                          labels_train: np.array,
                          features_valid: Optional[np.array] = None,
                          labels_valid: Optional[np.array] = None) -> float:
        model = self._get_model(
            svd_num_components=trial.suggest_int('svd_num_components', 1, max(self.num_embedding_features - 1, 1)),
            svm_penalty=trial.suggest_loguniform('svm_penalty', 0.1, 1e4)
        )
        model.fit(features_train, labels_train)

        if (features_valid is not None) and (labels_valid is not None):
            predictions_valid = model.predict(features_valid)
            score = f1_score(labels_valid, predictions_valid, average='macro')
        else:
            predictions_train = model.predict(features_train)
            score = f1_score(labels_train, predictions_train, average='macro')

        return score

