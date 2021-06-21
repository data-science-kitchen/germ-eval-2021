from flair.data import Corpus
from flair.embeddings import DocumentEmbeddings
import numpy as np
import os
from pathlib import Path
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import Callable, List, Optional, Tuple, Union


class FeatureExtractor:
    def __init__(self,
                 features: List[Callable],
                 document_embeddings: Optional[DocumentEmbeddings] = None) -> None:
        self.features = features
        self.document_embeddings = document_embeddings

    def compute_features(self,
                         corpus: Corpus,
                         save_file: Optional[Union[str, Path]] = None) -> Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]]:
        if save_file is not None and os.path.isfile(save_file):
            output = np.load(save_file)

            features_train = output['features_train']
            labels_train = output['labels_train']
            features_dev = output['features_dev']
            labels_dev = output['labels_dev']
        else:
            output = []

            if self.document_embeddings is not None:
                feature_dim = len(self.features) + self.document_embeddings.embedding_length
            else:
                feature_dim = len(self.features)

            for subset in [corpus.train, corpus.dev]:
                features = np.zeros((len(subset), feature_dim), dtype=np.float32)
                labels = np.zeros((len(subset), 3), dtype=np.float32)

                for sentence_idx, sentence in enumerate(subset):
                    for feature_idx, feature_func in enumerate(self.features):
                        features[sentence_idx, feature_idx] = feature_func(sentence.to_plain_string())

                    if self.document_embeddings is not None:
                        self.document_embeddings.embed(sentence)
                        features[sentence_idx, len(self.features):] = sentence.embedding.cpu().detach().numpy()

                    labels[sentence_idx, :] = np.asarray([x.value for x in sentence.labels])

                output.append((features, labels))

            features_train, labels_train = output[0]
            features_dev, labels_dev = output[1]

            np.savez(save_file,
                     features_train=features_train,
                     labels_train=labels_train,
                     features_dev=features_dev,
                     labels_dev=labels_dev)

        return (features_train, labels_train), (features_dev, labels_dev)


class SentimentModel:
    MAX_NUM_TOKENS = 512

    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = AutoModelForSequenceClassification.from_pretrained('oliverguhr/german-sentiment-bert')
        self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained('oliverguhr/german-sentiment-bert')
        self.clean_chars = re.compile(r'[^A-Za-züöäÖÜÄß ]', re.MULTILINE)
        self.clean_http_urls = re.compile(r'https*\S+', re.MULTILINE)
        self.clean_at_mentions = re.compile(r'@\S+', re.MULTILINE)

    def __call__(self, text: str) -> float:
        text = self._clean_text(text)

        input_ids = self.tokenizer.batch_encode_plus([text], padding=True, add_special_tokens=True)
        input_ids = torch.tensor(input_ids['input_ids'])
        input_ids = input_ids.to(self.device)

        if input_ids.shape[-1] > self.MAX_NUM_TOKENS:
            input_ids = input_ids[:, :self.MAX_NUM_TOKENS]

        with torch.no_grad():
            result = self.model(input_ids)

        return result.logits.squeeze().detach().cpu().numpy()

    def _clean_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = self.clean_http_urls.sub('', text)
        text = self.clean_at_mentions.sub('', text)
        text = self._replace_numbers(text)
        text = self.clean_chars.sub('', text)
        text = ' '.join(text.split())
        text = text.strip().lower()

        return text

    @staticmethod
    def _replace_numbers(text: str) -> str:
        return text.replace('0', ' null').replace('1', ' eins').replace('2', ' zwei').replace('3', ' drei') \
            .replace('4', ' vier').replace('5', ' fünf').replace('6', ' sechs').replace('7', ' sieben') \
            .replace('8', ' acht').replace('9', ' neun')


SENTIMENT_MODEL = SentimentModel()


def log_num_characters(text: str) -> float:
    """
    Args:
        text (str):
    """
    return np.log(len(text) + 1e-9)


def log_num_tokens(text: str) -> float:
    """
    Args:
        text (str):
    """
    return np.log(len(text.split()) + 1e-9)


def log_average_word_length(text: str) -> float:
    """
    Args:
        text (str):
    """
    word_lengths = [len(x) for x in text.split()]

    return np.log(np.mean(word_lengths) + 1e-9)


def log_word_length_std(text: str) -> float:
    """
    Args:
        text (str):
    """
    word_lengths = [len(x) for x in text.split()]

    return np.log(np.std(word_lengths) + 1e-9)


def positive_sentiment_logits(text: str) -> float:
    return SENTIMENT_MODEL(text)[0]


def negative_sentiment_logits(text: str) -> float:
    return SENTIMENT_MODEL(text)[1]


def neutral_sentiment_logits(text: str) -> float:
    return SENTIMENT_MODEL(text)[2]
