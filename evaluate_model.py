from dataset import GermEval2021
from features import *
import fire
from flair.embeddings import TransformerDocumentEmbeddings
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
from typing import Union


def main(corpus_file: Union[str, Path],
         random_seed: int = 42) -> None:
    results_list = []
    tasks = ['Toxic', 'Engaging', 'FactClaiming']

    document_embeddings = TransformerDocumentEmbeddings('bert-base-german-cased', fine_tune=False)

    features = [
        log_num_characters, log_average_word_length, log_word_length_std, positive_sentiment_logits,
        negative_sentiment_logits, neutral_sentiment_logits
    ]

    feature_extractor = FeatureExtractor(features, document_embeddings=document_embeddings)

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('classifier', MultiOutputClassifier(GaussianNB()))
    ])

    with tqdm(total=4) as progress_bar:
        for fold_idx in range(4):
            corpus = GermEval2021(corpus_file, fold=fold_idx)
            
            data_train, data_dev = feature_extractor.compute_features(corpus, save_file='features_fold{}.npz'.format(fold_idx))

            features_train, labels_train = data_train
            features_dev, labels_dev = data_dev

            pipeline.fit(features_train, labels_train)
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

        print('{:13s} === Accuracy: {:0.4f} +/- {:0.4f}, Precision: {:0.4f} +/- {:0.4f}, Recall: {:0.4f} +/- {:0.4f}, F1-Score: {:0.4f} +/- {:0.4f}'.format(
            task, accuracy_mean, accuracy_std, precision_mean, precision_std, recall_mean, recall_std, f1_score_mean, f1_score_std)
        )


if __name__ == '__main__':
    fire.Fire(main)
