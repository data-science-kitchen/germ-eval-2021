from dataset import process_corpus, GermEval2021
import fire
from flair.embeddings import TransformerDocumentEmbeddings, WordEmbeddings, DocumentPoolEmbeddings
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm
from typing import Union
from xgboost import XGBClassifier


def main(corpus_file: Union[str, Path],
         random_seed: int = 42) -> None:
    results_list = []
    tasks = ['Toxic', 'Engaging', 'FactClaiming']

    document_embeddings = TransformerDocumentEmbeddings('bert-base-german-cased', fine_tune=False)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MultiOutputClassifier(GaussianNB()))
    ])

    with tqdm(total=4) as progress_bar:
        for fold_idx in range(4):
            corpus = GermEval2021(corpus_file, fold=fold_idx)

            data_train, data_dev = process_corpus(corpus, document_embeddings)

            features_train, labels_train = data_train
            features_dev, labels_dev = data_dev

            pipeline.fit(features_train, labels_train)
            predictions = pipeline.predict(features_dev)

            for task_idx, task in enumerate(tasks):
                results_list.append({
                    'precision': precision_score(labels_dev[:, task_idx], predictions[:, task_idx]),
                    'recall': recall_score(labels_dev[:, task_idx], predictions[:, task_idx]),
                    'f1': f1_score(labels_dev[:, task_idx], predictions[:, task_idx]),
                    'fold_idx': fold_idx,
                    'task': task
                })

            progress_bar.update()

    results = pd.DataFrame(results_list)

    for task in tasks:
        precision_mean = results[results['task'] == task].precision.mean()
        precision_std = results[results['task'] == task].precision.std()
        recall_mean = results[results['task'] == task].recall.mean()
        recall_std = results[results['task'] == task].recall.std()
        f1_score_mean = results[results['task'] == task].f1.mean()
        f1_score_std = results[results['task'] == task].f1.std()

        print('{:13s} === Precision: {:0.4f} +/- {:0.4f}, Recall: {:0.4f} +/- {:0.4f}, F1-Score: {:0.4f} +/- {:0.4f}'.format(
            task, precision_mean, precision_std, recall_mean, recall_std, f1_score_mean, f1_score_std)
        )


if __name__ == '__main__':
    fire.Fire(main)
