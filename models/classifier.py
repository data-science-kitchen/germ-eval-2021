from preprocessing.features import Feature
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
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Optional, Union


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

        return (predictions.sum(axis=1) >= num_classifiers / 2.).astype(np.int64)


class GermEvalModel(ClassifierMixin):
    TASK_NAMES = ['Toxic', 'Engaging', 'FactClaiming']

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
                svd_num_components=study.best_params['svd_num_components'], lr_penalty=study.best_params['lr_penalty']
            )
            model.fit(features_train, labels_train)

            joblib.dump(model, save_file)

        self.model = model

    def predict(self, features: np.array) -> np.array:
        if self.model is not None:
            return self.model.predict(features)
        else:
            raise ValueError('No trained model found. Please run .fit() first.')

    def get_feature_importance(self,
                               top_k: int = 10) -> Dict:
        feature_names = self._get_feature_names()
        feature_type = self._get_feature_type()

        numerical_features = [feature_names[idx] for idx, x in enumerate(feature_type) if x == 'numerical']
        num_numerical_features = len(numerical_features)

        estimators = self.model.steps[-1][1].estimators_

        output = {}

        for estimator, task in zip(estimators, self.TASK_NAMES):
            feature_importance = np.abs(estimator.coef_.squeeze())

            embedding_dim = feature_importance.shape[-1] - num_numerical_features
            reduced_feature_names = numerical_features + embedding_dim * ['EmbeddingFeature']

            sort_idx = feature_importance.argsort()
            feature_importance = feature_importance[sort_idx]
            reduced_feature_names = [reduced_feature_names[x] for x in sort_idx]
            feature_importance, reduced_feature_names = feature_importance[-top_k:], reduced_feature_names[-top_k:]

            output[task] = [feature_importance, reduced_feature_names]

        return output

    def _get_feature_names(self) -> List[str]:
        feature_names = []

        for feature_func in self.feature_funcs:
            feature_names += feature_func.dim * [feature_func.__class__.__name__]

        return feature_names

    def _get_feature_type(self) -> List[str]:
        feature_type = []

        for feature_func in self.feature_funcs:
            feature_type += feature_func.dim * [feature_func.type]

        return feature_type

    def _get_model(self,
                   svd_num_components: Optional[int] = None,
                   lr_penalty: Optional[float] = 1.0):
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
            ('classifier', MultiOutputClassifier(
                LogisticRegression(C=lr_penalty, penalty='l1', max_iter=300, solver='liblinear', tol=0.1), n_jobs=-1
            ))
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
            lr_penalty=trial.suggest_loguniform('lr_penalty', 0.1, 1e4)
        )
        model.fit(features_train, labels_train)

        if (features_valid is not None) and (labels_valid is not None):
            predictions_valid = model.predict(features_valid)
            score = f1_score(labels_valid, predictions_valid, average='macro')
        else:
            predictions_train = model.predict(features_train)
            score = f1_score(labels_train, predictions_train, average='macro')

        return score
