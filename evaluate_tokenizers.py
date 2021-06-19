from dataset import process_corpus, GermEval2021
import fire
from flair.embeddings import TransformerDocumentEmbeddings
from flair.tokenization import SegtokTokenizer, SpaceTokenizer, SpacyTokenizer
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

        tokenizers = [SpaceTokenizer(), SegtokTokenizer(), SpacyTokenizer('de_dep_news_trf')]

        with tqdm(total=4 * len(tokenizers)) as progress_bar:
            for tokenizer in tokenizers:
                for fold_idx in range(4):
                    corpus = GermEval2021(corpus_file, fold=fold_idx, tokenizer=tokenizer)

                    document_embeddings = TransformerDocumentEmbeddings('bert-base-german-cased', fine_tune=False)
                    data_train, data_dev = process_corpus(corpus, document_embeddings)

                    classifier = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', MultiOutputClassifier(XGBClassifier(use_label_encoder=False,
                                                                           random_state=random_seed)))
                    ])
                    classifier.fit(data_train[0], data_train[1])

                    predictions = classifier.predict(data_dev[0])

                    for task_idx, task_name in enumerate(tasks):
                        results_list.append({
                            'tokenizer': tokenizer.__class__.__name__,
                            'precision': precision_score(data_dev[1], predictions, average=None)[task_idx],
                            'recall': recall_score(data_dev[1], predictions, average=None)[task_idx],
                            'f1': f1_score(data_dev[1], predictions, average=None)[task_idx],
                            'fold_idx': fold_idx,
                            'task': task_name
                        })

                    progress_bar.update()

        results = pd.DataFrame(results_list)
        results.to_json(results_file)

    sns.boxplot(x='tokenizer', y='f1', hue='task', data=results)
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)
