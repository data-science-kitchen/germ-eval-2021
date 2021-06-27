import abc
from flair.data import Corpus, Sentence
from flair.embeddings import DocumentEmbeddings
from flair.tokenization import SegtokTokenizer, Tokenizer
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
from spellchecker import SpellChecker
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import Callable, List, Optional, Tuple, Union
from tqdm import tqdm


class Feature(abc.ABC):
    @property
    @abc.abstractmethod
    def dim(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def type(self) -> str:
        pass

    def __call__(self, text: str) -> Union[float, np.array]:
        pass


class FeatureExtractor:
    def __init__(self,
                 feature_funcs: List[Feature]) -> None:
        self.feature_funcs = feature_funcs

    def get_features(self,
                     dataset_file: Union[str, Path],
                     save_file: Optional[Union[str, Path]] = None,
                     show_progress_bar: bool = False) -> Tuple[np.array, np.array]:
        if save_file is not None and os.path.isfile(save_file):
            file = np.load(save_file, allow_pickle=True)
            features, labels = file['features'], file['labels']
        else:
            data_frame = pd.read_csv(dataset_file, header=0)

            num_documents = len(data_frame)
            feature_dim = self._get_feature_dim()

            has_labels = self._check_has_labels(data_frame)

            features = np.zeros((num_documents, feature_dim))
            labels = np.zeros((num_documents, 3), dtype=np.int64) if has_labels else None

            for row in tqdm(data_frame.iterrows(),
                            desc='Computing features', total=len(data_frame), disable=not show_progress_bar):
                row_idx, text = row[0], row[1][1]

                feature_idx = 0
                for feature_func in self.feature_funcs:
                    features[row_idx, feature_idx] = feature_func(text)
                    feature_idx += feature_func.dim

                if has_labels:
                    labels[row_idx, :] = row[1][2:].to_numpy()

            np.savez(save_file, features=features, labels=labels)

        return features, labels

    @staticmethod
    def _check_has_labels(data_frame: pd.DataFrame) -> bool:
        return 'Sub1_Toxic' in data_frame.columns and \
               'Sub2_Engaging' in data_frame.columns and \
               'Sub3_FactClaiming' in data_frame.columns

    def _get_feature_dim(self):
        feature_dim = 0

        for feature_func in self.feature_funcs:
            feature_dim += feature_func.dim

        return feature_dim


class NumCharacters(Feature):
    def __init__(self,
                 apply_log: bool = False) -> None:
        self.apply_log = apply_log

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return 'numerical'

    def __call__(self, text: str) -> float:
        if self.apply_log:
            return np.log(len(text) + 1e-9)
        else:
            return float(len(text))


class NumTokens(Feature):
    def __init__(self,
                 apply_log: bool = False) -> None:
        self.apply_log = apply_log

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return 'numerical'

    def __call__(self, text: str) -> float:
        if self.apply_log:
            return np.log(len(text.split()) + 1e-9)
        else:
            return float(len(text.split()))


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


def spelling_mistake_log_odds(text: str) -> float:
    """
    Args:
        text (str):
    """
    spell = SpellChecker(language='de')

    tokens = text.split()
    mistakes = spell.unknown(tokens)

    return np.log(len(mistakes) + 1e-9) - np.log(len(tokens) + 1e-9)
