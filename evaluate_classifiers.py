from cleanlab.classification import LearningWithNoisyLabels
from dataset import process_corpus, GermEval2021
import fire
from flair.embeddings import TransformerDocumentEmbeddings
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm
from typing import Union
from xgboost import XGBClassifier


def main(corpus_file: Union[str, Path],
         results_file: Union[str, Path] = 'results.json',
         random_seed: int = 42) -> None:
    if os.path.isfile(results_file):
        results = pd.read_json(results_file)
    else:
        results_list = []
        tasks = ['Toxic', 'Engaging', 'FactClaiming']

        classifiers = [LogisticRegression(max_iter=500, random_state=random_seed),
                       LearningWithNoisyLabels(LogisticRegression(max_iter=500, random_state=random_seed)),
                       GaussianNB(),
                       LearningWithNoisyLabels(GaussianNB()),
                       LinearSVC(max_iter=3000, random_state=random_seed),
                       XGBClassifier(use_label_encoder=False, random_state=random_seed)]
        classifier_descriptions = [
            'Logistic Regression',
            'Logistic Regression + Cleanlab',
            'Naive Bayes',
            'Naive Bayes + Cleanlab',
            'Linear SVM',
            'XGBoost'
        ]

        with tqdm(total=4 * len(classifiers)) as progress_bar:
            for fold_idx in range(4):
                corpus = GermEval2021(corpus_file, fold=fold_idx)

                document_embeddings = TransformerDocumentEmbeddings('bert-base-german-cased', fine_tune=False)
                data_train, data_dev = process_corpus(corpus, document_embeddings)
                
                features_train, labels_train = data_train
                
                scaler = StandardScaler()
                features_train = scaler.fit_transform(features_train)

                features_dev, labels_dev = data_dev
                features_dev = scaler.transform(features_dev)

                for cls_idx, classifier in enumerate(classifiers):
                    for task_idx, task in enumerate(tasks):
                        classifier.fit(features_train, labels_train[:, task_idx])
                        predictions = classifier.predict(features_dev)

                        results_list.append({
                            'classifier': classifier_descriptions[cls_idx],
                            'precision': precision_score(labels_dev[:, task_idx], predictions),
                            'recall': recall_score(labels_dev[:, task_idx], predictions),
                            'f1': f1_score(labels_dev[:, task_idx], predictions),
                            'fold_idx': fold_idx,
                            'task': task
                        })

                    progress_bar.update()

        results = pd.DataFrame(results_list)
        results.to_json(results_file)

    sns.boxplot(x='classifier', y='f1', hue='task', data=results)
    plt.grid(True)
    plt.ylim((0, 1))
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)
