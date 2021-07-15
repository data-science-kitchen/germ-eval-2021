import abc
from advertools import extract_emoji, stopwords
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, WordEmbeddings
from flair.embeddings import TransformerDocumentEmbeddings
from models.adhominem import AdHominem
from nltk.corpus import stopwords
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import re
import language_tool_python
from somajo import SoMaJo
import tensorflow.compat.v1 as tf
from textblob_de import TextBlobDE as TextBlob
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

tf.disable_v2_behavior()


class Feature(abc.ABC):
    @property
    @abc.abstractmethod
    def dim(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def type(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def is_trainable(self) -> bool:
        pass

    @property
    def feature_names(self) -> np.array:
        return self.dim * [self.__class__.__name__]

    def __call__(self, text: str) -> Union[float, np.array]:
        pass

    def fit(self, text: List[str], labels: Optional[np.array] = None) -> None:
        if not self.is_trainable:
            raise ValueError('Feature is not trainable')
        else:
            pass


class AverageEmojiRepetition(Feature):
    def __init__(self) -> None:
        pass

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return 'numerical'

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        if len(extract_emoji(text)['top_emoji']) > 0:
            average_emoji_counts = np.array(extract_emoji(text)['top_emoji'])[:, 1].astype(int).mean()
        else:
            average_emoji_counts = 0
        return np.log(average_emoji_counts + 1e-9)


class NumUserAdressed(Feature):
    def __init__(self) -> None:
        pass
    
    @property
    def dim(self):
        return 1
    
    @property
    def type(self):
        return 'numerical'

    @property
    def is_trainable(self) -> bool:
        return False
    
    def __call__(self, text: str) -> float:
        user_adressed = np.sum([token == '@user' for token in text.lower().split()])
        return np.log(user_adressed + 1e-9)


class NumMediumAdressed(Feature):
    def __init__(self) -> None:
        pass
    
    @property
    def dim(self):
        return 1
    
    @property
    def type(self):
        return 'numerical'

    @property
    def is_trainable(self) -> bool:
        return False
    
    def __call__(self, text: str) -> float:
        medium_adressed = np.sum([token == '@medium' for token in text.lower().split()])
        return np.log(medium_adressed + 1e-9)


class NumReferences(Feature):
    def __init__(self) -> None:
        pass
    
    @property
    def dim(self):
        return 1
    
    @property
    def type(self):
        return 'numerical'

    @property
    def is_trainable(self) -> bool:
        return False
    
    def __call__(self, text: str) -> float:
        refs = re.findall(r'http*\S+', text)
        return np.log(len(refs) + 1e-9)


class ExclamationMarkRatio(Feature):
    def __init__(self) -> None:
        pass
    
    @property
    def dim(self):
        return 1
    
    @property
    def type(self):
        return 'numerical'

    @property
    def is_trainable(self) -> bool:
        return False
    
    def __call__(self, text: str) -> float:
        exclamation_mark_ratio = np.sum([char=='!' for char in text]) / len(text)
        return exclamation_mark_ratio * 10


class StopwordRatio(Feature):
    def __init__(self) -> None:
        pass
    
    @property
    def dim(self):
        return 1
    
    @property
    def type(self):
        return 'numerical'

    @property
    def is_trainable(self) -> bool:
        return False
    
    def __call__(self, text: str) -> float:
        tokens = text.lower().split()
        ratio = np.sum([token in stopwords.words('german') for token in tokens]) / len(tokens)

        return np.log(ratio + 0.5 + 1e-9)


class NumCharacters(Feature):
    def __init__(self) -> None:
        pass

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return 'numerical'

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        return np.log(len(text) + 1e-9)


class NumTokens(Feature):
    def __init__(self) -> None:
        pass

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return 'numerical'

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        return np.log(len(text.split()) + 1e-9)


class AverageTokenLength(Feature):
    def __init__(self) -> None:
        pass

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return 'numerical'

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        word_lengths = [len(x) for x in text.split()]
        return np.log(np.mean(word_lengths) + 1e-9)


class TokenLengthStandardDeviation(Feature):
    def __init__(self) -> None:
        pass

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return 'numerical'

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        word_lengths = [len(x) for x in text.split()]
        return np.log(np.std(word_lengths) + 1e-9)


class SpellingMistakes(Feature):
    def __init__(self) -> None:
        self.spell_checker = language_tool_python.LanguageTool('de')

    @property
    def dim(self):
        return 17

    @property
    def type(self):
        return 'numerical'

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def feature_names(self) -> List[str]:
        names = [
            'MISC',
            'RECOMMENDED_SPELLING',
            'TYPOGRAPHY',
            'PUNCTUATION',
            'GRAMMAR',
            'CASING',
            'SUPPORT_PUNCTUATION',
            'COLLOQUIALISMS',
            'COMPOUNDING',
            'CONFUSED_WORDS',
            'REDUNDANCY',
            'TYPOS',
            'STYLE',
            'PROPER_NOUNS',
            'IDIOMS',
            'DE_DOUBLE_PUNCTUATION',
            'DOUBLE_EXCLAMATION_MARK'
        ]
        return names

    def __call__(self, text: str) -> np.array:
        mistakes = self.spell_checker.check(text)

        output = np.zeros((1, self.dim))

        if len(mistakes) > 0:
            for mistake in mistakes:
                if mistake.category == 'MISC':
                    output[0, 0] += 1
                elif mistake.category == 'EMPFOHLENE_RECHTSCHREIBUNG':
                    output[0, 1] += 1
                elif mistake.category == 'TYPOGRAPHY' and \
                        mistake.ruleId not in ['TYPOGRAFISCHE_ANFUEHRUNGSZEICHEN',
                                               'FALSCHE_VERWENDUNG_DES_BINDESTRICHS',
                                               'AKZENT_STATT_APOSTROPH', 'MULTIPLICATION_SIGN']:
                    output[0, 2] += 1
                elif mistake.category == 'PUNCTUATION' and \
                        mistake.ruleId not in ['EINHEIT_LEERZEICHEN', 'ZEICHENSETZUNG_DIREKTE_REDE']:
                    output[0, 3] += 1
                elif mistake.category == 'GRAMMAR':
                    output[0, 4] += 1
                elif mistake.category == 'CASING':
                    output[0, 5] += 1
                elif mistake.category == 'HILFESTELLUNG_KOMMASETZUNG':
                    output[0, 6] += 1
                elif mistake.category == 'COLLOQUIALISMS':
                    output[0, 7] += 1
                elif mistake.category == 'COMPOUNDING':
                    output[0, 8] += 1
                elif mistake.category == 'CONFUSED_WORDS':
                    output[0, 9] += 1
                elif mistake.category == 'REDUNDANCY':
                    output[0, 10] += 1
                elif mistake.category == 'TYPOS':
                    output[0, 11] += 1
                elif mistake.category == 'STYLE':
                    output[0, 12] += 1
                elif mistake.category == 'PROPER_NOUNS':
                    output[0, 13] += 1
                elif mistake.category == 'IDIOMS':
                    output[0, 14] += 1
                elif mistake.category == 'PUNCTUATION' and mistake.ruleId == 'DE_DOUBLE_PUNCTUATION':
                    output[0, 15] += 1
                elif mistake.category == 'PUNCTUATION' and mistake.ruleId == 'DOPPELTES_AUSRUFEZEICHEN':
                    output[0, 16] += 1

        return np.log(output + 1e-9) - np.log(len(text.split()) + 1e-9)


class DocumentEmbeddingsFastTextPool(Feature):
    def __init__(self) -> None:
        word_embeddings = WordEmbeddings('de')
        self.document_embeddings = DocumentPoolEmbeddings([word_embeddings])

    @property
    def dim(self):
        return 300

    @property
    def type(self):
        return 'embedding'

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        sentence = Sentence(text)
        self.document_embeddings.embed(sentence)

        return sentence.embedding.detach().cpu().numpy()


class DocumentEmbeddingsBERT(Feature):
    def __init__(self) -> None:
        self.embeddings = TransformerDocumentEmbeddings('bert-base-german-cased', fine_tune=False)

    @property
    def dim(self):
        return 768

    @property
    def type(self):
        return 'embedding'

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        sentence = Sentence(text)
        self.embeddings.embed(sentence)

        return sentence.embedding.detach().cpu().numpy()


class SentimentTextBlob(Feature):
    def __init__(self) -> None:
        pass

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return 'numerical'

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        blob = TextBlob(text)

        return blob.sentiment.polarity


class SentimentBERT(Feature):
    MAX_NUM_TOKENS = 512

    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = AutoModelForSequenceClassification.from_pretrained('oliverguhr/german-sentiment-bert')
        self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained('oliverguhr/german-sentiment-bert')
        self.clean_chars = re.compile(r'[^A-Za-züöäÖÜÄß ]', re.MULTILINE)
        self.clean_http_urls = re.compile(r'https*\S+', re.MULTILINE)
        self.clean_at_mentions = re.compile(r'@\S+', re.MULTILINE)

    @property
    def dim(self):
        return 3

    @property
    def type(self):
        return 'numerical'

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> np.array:
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


class WritingStyleEmbeddings(Feature):
    def __init__(self) -> None:
        self.tokenizer = SoMaJo(language='en_PTB', split_sentences=True)  # "de_CMC"

        tf.reset_default_graph()

        with open(os.path.join('adhominem'), 'rb') as f:
            parameters = pickle.load(f)

        with tf.variable_scope('AdHominem'):
            self.adhominem = AdHominem(hyper_parameters=parameters['hyper_parameters'],
                                       theta_init=parameters['theta'],
                                       theta_E_init=parameters['theta_E'])

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    @property
    def dim(self):
        return 100

    @property
    def type(self):
        return 'embedding'

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> np.array:
        processed_text = self.preprocess_doc(text, self.tokenizer)
        embeddings = self.make_inference(processed_text, self.adhominem, self.session).squeeze()

        return embeddings

    @staticmethod
    def preprocess_doc(doc, tokenizer):
        sentences = tokenizer.tokenize_text([doc])
        doc_new = ''
        for sentence in sentences:
            for token in sentence:
                doc_new += token.text + ' '
        doc_new = doc_new.strip()

        return doc_new

    @staticmethod
    def make_inference(doc, adhominem, sess):
        emb = adhominem.inference([doc], sess)

        return emb


class FeatureExtractor:
    def __init__(self,
                 feature_funcs: List[Feature]) -> None:
        self.feature_funcs = feature_funcs

    def get_features(self,
                     dataset_file: Union[str, Path],
                     save_file: Optional[Union[str, Path]] = None,
                     train: bool = False,
                     show_progress_bar: bool = False) -> Tuple[np.array, np.array]:
        if save_file is not None and os.path.isfile(save_file):
            file = np.load(save_file, allow_pickle=True)
            features, labels = file['features'], file['labels']
        else:
            data_frame = pd.read_csv(dataset_file, header=0)

            num_documents = len(data_frame)
            feature_dim = self._get_feature_dim()

            has_labels = self._check_has_labels(data_frame)

            if train:
                train_text = data_frame.iloc[:, 1].tolist()
                train_labels = data_frame.iloc[:, 2:].to_numpy() if has_labels else None

                for feature_func in self.feature_funcs:
                    if feature_func.is_trainable:
                        feature_func.fit(train_text, train_labels)

            features = np.zeros((num_documents, feature_dim))
            labels = np.zeros((num_documents, 3), dtype=np.int64) if has_labels else None

            for row in tqdm(data_frame.iterrows(),
                            desc='Computing features', total=len(data_frame), disable=not show_progress_bar):
                row_idx, text = row[0], row[1][1]

                feature_idx = 0
                for feature_func in self.feature_funcs:
                    features[row_idx, feature_idx:feature_idx+feature_func.dim] = feature_func(text)
                    feature_idx += feature_func.dim

                if has_labels:
                    labels[row_idx, :] = row[1][2:].to_numpy()

            if save_file is not None:
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
