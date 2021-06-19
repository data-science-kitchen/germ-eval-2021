from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import DocumentEmbeddings
import numpy as np
import pandas as pd
from pathlib import Path
import os
from typing import Tuple, Union


class GermEval2021(CSVClassificationCorpus):
    def __init__(self,
                 base_path: Union[str, Path],
                 fold: int = 0,
                 **corpusargs):
        data_frame = pd.read_csv(base_path, header=0, sep=';')
        data_frame = data_frame.drop(columns=['comment_id'])

        fold_dir = Path(os.path.dirname(base_path), 'fold_{}'.format(fold))
        fold_dir.mkdir(parents=True, exist_ok=True)

        dev_fold_indices = list(data_frame[data_frame.fold == fold].index)

        training_data = data_frame.drop(data_frame.index[dev_fold_indices]).reset_index(drop=True).drop(columns=['fold'])
        training_data.to_csv(fold_dir / 'train.csv', index=False)

        dev_data = data_frame.iloc[data_frame.index[dev_fold_indices]].reset_index(drop=True).drop(columns=['fold'])
        dev_data.to_csv(fold_dir / 'dev.csv', index=False)
        
        super(GermEval2021, self).__init__(
            data_folder=fold_dir,
            column_name_map={0: 'text', 1: 'label_toxic', 2: 'label_engaging', 3: 'label_fact_claiming'},
            skip_header=True,
            train_file='train.csv',
            dev_file='dev.csv',
            **corpusargs
        )


def process_corpus(corpus: Corpus,
                   document_embeddings: DocumentEmbeddings) -> Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]]:
    embeddings_train = []
    labels_train = []

    for sentence in corpus.train:
        document_embeddings.embed(sentence)

        embeddings_train.append(sentence.embedding.cpu().detach().numpy())
        labels_train.append(np.asarray([float(x.value) for x in sentence.labels]))

    embeddings_train = np.asarray(embeddings_train)
    labels_train = np.asarray(labels_train)

    embeddings_dev = []
    labels_dev = []

    for sentence in corpus.dev:
        document_embeddings.embed(sentence)

        embeddings_dev.append(sentence.embedding.cpu().detach().numpy())
        labels_dev.append(np.asarray([float(x.value) for x in sentence.labels]))

    embeddings_dev = np.asarray(embeddings_dev)
    labels_dev = np.asarray(labels_dev)

    return (embeddings_train, labels_train), (embeddings_dev, labels_dev)
