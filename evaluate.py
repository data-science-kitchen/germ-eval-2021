import copy
from dataset import GermEval2021
from features import *
import fire
from flair.embeddings import TransformerDocumentEmbeddings
from models import GermEvalMLP
import pandas as pd
from pathlib import Path
import optuna
from scipy.sparse import lil_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from skorch import NeuralNet
from tqdm import tqdm
import torch
import torch.nn as nn
from typing import Optional, Union


def objective_function(trial: optuna.Trial,
                       features_train: np.array,
                       labels_train: np.array,
                       features_dev: np.array,
                       labels_dev: np.array,
                       feature_dim: int,
                       embedding_dim: int,
                       device='cpu') -> Tuple[float, float, float]:
    model = NeuralNet(
        GermEvalMLP(feature_dim=feature_dim,
                    embedding_dim=embedding_dim,
                    hidden_dim=trial.suggest_categorical('hidden_dim', [4, 8, 16, 32, 64]),
                    dropout_rate=trial.suggest_uniform('dropout_rate', 0.0, 0.9)),
        max_epochs=trial.suggest_categorical('max_epochs', [50, 100, 150, 200, 250, 300]),
        criterion=nn.BCELoss,
        lr=trial.suggest_loguniform('learning_rate', 1e-5, 0.01),
        optimizer=torch.optim.SGD,
        batch_size=trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128]),
        verbose=0,
        device=device
    )

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('classifier', copy.deepcopy(model))
    ])

    pipeline.fit(features_train, labels_train)

    classification_threshold = trial.suggest_uniform('classification_threshold', 0.05, 0.95)
    predicted_labels = (pipeline.predict(features_dev) > classification_threshold).astype(int)

    score_toxic = f1_score(labels_dev[:, 0], predicted_labels[:, 0], average='macro')
    score_engaging = f1_score(labels_dev[:, 1], predicted_labels[:, 1], average='macro')
    score_fact_claiming = f1_score(labels_dev[:, 2], predicted_labels[:, 2], average='macro')

    return score_toxic, score_engaging, score_fact_claiming


def main(corpus_file: Union[str, Path],
         tmp_dir: Optional[Union[str, Path]] = None) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            else:
                save_file = None
            data_train, data_dev = feature_extractor.compute_features(corpus, save_file=save_file)

            features_train, labels_train = data_train
            features_dev, labels_dev = data_dev

            sampler = optuna.samplers.NSGAIISampler()
            study = optuna.create_study(directions=['maximize', 'maximize', 'maximize'], sampler=sampler)

            study.optimize(lambda x: objective_function(x,
                                                        features_train,
                                                        labels_train,
                                                        features_dev,
                                                        labels_dev,
                                                        len(features),
                                                        document_embeddings.embedding_length,
                                                        device=device),
                           n_trials=250,
                           gc_after_trial=True,
                           n_jobs=-1)

            print(study.best_params)

            model = NeuralNet(
                GermEvalMLP(feature_dim=len(features),
                            embedding_dim=document_embeddings.embedding_length,
                            hidden_dim=study.best_params['hidden_dim'],
                            dropout_rate=study.best_params['dropout_rate']),
                max_epochs=study.best_params['max_epochs'],
                criterion=nn.BCELoss,
                lr=study.best_params['learning_rate'],
                optimizer=torch.optim.SGD,
                batch_size=study.best_params['batch_size'],
                device=device
            )

            pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('classifier', copy.deepcopy(model))
            ])

            pipeline.fit(features_train, labels_train)
            predicted_labels = (pipeline.predict(features_dev) > study.best_params['classification_threshold']).astype(int)

            if isinstance(predicted_labels, lil_matrix):
                predicted_labels = predicted_labels.toarray()

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


if __name__ == '__main__':
    fire.Fire(main)
