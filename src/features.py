import abc
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import language_tool_python
import nltk
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import torch
from advertools import extract_emoji
from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings, TransformerDocumentEmbeddings, WordEmbeddings
from nltk.corpus import stopwords
from skops.io import load
from somajo import SoMaJo
from textblob_de import TextBlobDE as TextBlob
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.adhominem import AdHominem

tf.disable_v2_behavior()

try:
    stopwords.words("german")
except LookupError:
    nltk.download("stopwords")


class Feature(abc.ABC):
    """
    An abstract base class for defining features used in machine learning models.

    This class serves as a blueprint for creating different types of features that can be extracted from text data. Each
    subclass of `Feature` should implement the abstract properties and methods to define the dimensionality, type, and
    trainability of the feature. Additionally, the `__call__` method should be overridden to define how the feature is
    extracted from a given text.

    Methods
    -------
    __call__(text: str) -> Union[float, np.array]
        Abstract method to extract the feature from a given text. Should be implemented in subclasses.
    fit(text: List[str], labels: Optional[np.array] = None) -> None
        Method to train the feature extraction process, if applicable. Raises a ValueError if the feature is not
        trainable.
    """

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
            raise ValueError("Feature is not trainable")
        else:
            pass


class AverageEmojiRepetition(Feature):
    """
    A feature that calculates the average repetition of emojis in a given text.

    This class is a specific implementation of the `Feature` abstract base class. It extracts the average repetition
    count of emojis in the text and returns it as a numerical feature. The result is logarithmically scaled to prevent
    skewness caused by large counts.

    Methods
    -------
    dim() -> int
        Returns the dimensionality of the feature, which is 1.
    type() -> str
        Returns the type of the feature, which is "numerical".
    is_trainable() -> bool
        Indicates whether the feature is trainable. Always returns False.
    __call__(text: str) -> float
        Extracts the average emoji repetition count from the text and returns its logarithmically scaled value.
    """

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return "numerical"

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        if len(extract_emoji(text)["top_emoji"]) > 0:
            average_emoji_counts = np.array(extract_emoji(text)["top_emoji"])[:, 1].astype(int).mean()
        else:
            average_emoji_counts = 0
        return np.log(average_emoji_counts + 1e-9)


class NumUserAdressed(Feature):
    """
    A feature that calculates the number of times users are addressed in a given text.

    This class counts the occurrences of the placeholder "@user" in the text, which typically represents user mentions,
    and returns the count as a numerical feature. The result is logarithmically scaled to handle large counts.

    Methods
    -------
    dim() -> int
        Returns the dimensionality of the feature, which is 1.
    type() -> str
        Returns the type of the feature, which is "numerical".
    is_trainable() -> bool
        Indicates whether the feature is trainable. Always returns False.
    __call__(text: str) -> float
        Extracts the number of times users are addressed in the text and returns its logarithmically scaled value.
    """

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return "numerical"

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        user_adressed = np.sum([token == "@user" for token in text.lower().split()])
        return np.log(user_adressed + 1e-9)


class NumMediumAdressed(Feature):
    """
    A feature that calculates the number of times a medium (e.g., news outlet) is addressed in a given text.

    This class counts the occurrences of the placeholder "@medium" in the text, which typically represents mentions of a
    medium or organization, and returns the count as a numerical feature. The result is logarithmically scaled to handle
    large counts.

    Methods
    -------
    dim() -> int
        Returns the dimensionality of the feature, which is 1.
    type() -> str
        Returns the type of the feature, which is "numerical".
    is_trainable() -> bool
        Indicates whether the feature is trainable. Always returns False.
    __call__(text: str) -> float
        Extracts the number of times a medium is addressed in the text and returns its logarithmically scaled value.
    """

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return "numerical"

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        medium_adressed = np.sum([token == "@medium" for token in text.lower().split()])
        return np.log(medium_adressed + 1e-9)


class NumReferences(Feature):
    """
    A feature that calculates the number of references (e.g., URLs) in a given text.

    This class counts the occurrences of URLs in the text using a regular expression and returns the count as a
    numerical feature. The result is logarithmically scaled to handle large counts.

    Methods
    -------
    dim() -> int
        Returns the dimensionality of the feature, which is 1.
    type() -> str
        Returns the type of the feature, which is "numerical".
    is_trainable() -> bool
        Indicates whether the feature is trainable. Always returns False.
    __call__(text: str) -> float
        Extracts the number of references (URLs) in the text and returns its logarithmically scaled value.
    """

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return "numerical"

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        refs = re.findall(r"http*\S+", text)
        return np.log(len(refs) + 1e-9)


class ExclamationMarkRatio(Feature):
    """
    A feature that calculates the ratio of exclamation marks in a given text.

    This class computes the proportion of characters in the text that are exclamation marks (`!`), and returns this
    ratio as a numerical feature.

    Methods
    -------
    dim() -> int
        Returns the dimensionality of the feature, which is 1.
    type() -> str
        Returns the type of the feature, which is "numerical".
    is_trainable() -> bool
        Indicates whether the feature is trainable. Always returns False.
    __call__(text: str) -> float
        Computes the ratio of exclamation marks in the text.
    """

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return "numerical"

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        exclamation_mark_ratio = np.sum([char == "!" for char in text]) / len(text)
        return exclamation_mark_ratio


class StopwordRatio(Feature):
    """
    A feature that calculates the ratio of stopwords in a given text.

    This class computes the proportion of tokens in the text that are stopwords, using a predefined list of German
    stopwords, and returns this ratio as a numerical feature. The result is logarithmically scaled with an offset to
    handle edge cases.

    Methods
    -------
    dim() -> int
        Returns the dimensionality of the feature, which is 1.
    type() -> str
        Returns the type of the feature, which is "numerical".
    is_trainable() -> bool
        Indicates whether the feature is trainable. Always returns False.
    __call__(text: str) -> float
        Computes the ratio of stopwords in the text and returns the logarithmically scaled value.
    """

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return "numerical"

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        tokens = text.lower().split()
        ratio = np.sum([token in stopwords.words("german") for token in tokens]) / len(tokens)

        return np.log(ratio + 1e-9)


class NumCharacters(Feature):
    """
    A feature that calculates the number of characters in a given text.

    This class computes the total number of characters in the text and returns this count as a logarithmically scaled
    numerical feature.

    Methods
    -------
    dim() -> int
        Returns the dimensionality of the feature, which is 1.
    type() -> str
        Returns the type of the feature, which is "numerical".
    is_trainable() -> bool
        Indicates whether the feature is trainable. Always returns False.
    __call__(text: str) -> float
        Computes the logarithmically scaled number of characters in the text.
    """

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return "numerical"

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        return np.log(len(text) + 1e-9)


class NumWords(Feature):
    """
    A feature that calculates the number of words in a given text.

    This class computes the total number of words in the text and returns this count as a logarithmically scaled
    numerical feature.

    Methods
    -------
    dim() -> int
        Returns the dimensionality of the feature, which is 1.
    type() -> str
        Returns the type of the feature, which is "numerical".
    is_trainable() -> bool
        Indicates whether the feature is trainable. Always returns False.
    __call__(text: str) -> float
        Computes the logarithmically scaled number of words in the text.
    """

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return "numerical"

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        return np.log(len(text.split()) + 1e-9)


class AverageWordLength(Feature):
    """
    A feature that calculates the average word length in a given text.

    This class computes the average word length in the text and returns it as a logarithmically scaled numerical
    feature.

    Methods
    -------
    dim() -> int
        Returns the dimensionality of the feature, which is 1.
    type() -> str
        Returns the type of the feature, which is "numerical".
    is_trainable() -> bool
        Indicates whether the feature is trainable. Always returns False.
    __call__(text: str) -> float
        Computes the logarithmically scaled average word length in the text.
    """

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return "numerical"

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        word_lengths = [len(x) for x in text.split()]
        return np.log(np.mean(word_lengths) + 1e-9)


class WordLengthStandardDeviation(Feature):
    """
    A feature that calculates the standard deviation of word lengths in a given text.

    This class computes the standard deviation of word lengths in the text and returns this value as a logarithmically
    scaled numerical feature.

    Methods
    -------
    dim() -> int
        Returns the dimensionality of the feature, which is 1.
    type() -> str
        Returns the type of the feature, which is "numerical".
    is_trainable() -> bool
        Indicates whether the feature is trainable. Always returns False.
    __call__(text: str) -> float
        Computes the logarithmically scaled standard deviation of word lengths in the text.
    """

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return "numerical"

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        word_lengths = [len(x) for x in text.split()]
        return np.log(np.std(word_lengths) + 1e-9)


class SpellingMistakes(Feature):
    """
    A feature that counts various types of spelling and grammatical mistakes in a given text.

    This class uses the `language_tool_python` library to detect spelling and grammatical mistakes and categorizes them
    into different types. The class returns the counts of these mistake types as a logarithmically scaled numerical
    feature.

    Attributes
    ----------
    spell_checker : language_tool_python.LanguageTool
        An instance of the `LanguageTool` class for German language, used to check text for mistakes.

    Methods
    -------
    __init__() -> None
        Initializes the SpellingMistakes feature with a German language spell checker.
    dim() -> int
        Returns the dimensionality of the feature, which is 17.
    type() -> str
        Returns the type of the feature, which is "numerical".
    is_trainable() -> bool
        Indicates whether the feature is trainable. Always returns False.
    feature_names() -> List[str]
        Returns a list of feature names corresponding to the types of mistakes detected.
    __call__(text: str) -> np.array
        Computes the logarithmically scaled counts of various types of spelling and grammatical mistakes in the text.
    """

    def __init__(self) -> None:
        self.spell_checker = language_tool_python.LanguageTool("de")

    @property
    def dim(self):
        return 17

    @property
    def type(self):
        return "numerical"

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def feature_names(self) -> List[str]:
        names = [
            "MISC",
            "RECOMMENDED_SPELLING",
            "TYPOGRAPHY",
            "PUNCTUATION",
            "GRAMMAR",
            "CASING",
            "SUPPORT_PUNCTUATION",
            "COLLOQUIALISMS",
            "COMPOUNDING",
            "CONFUSED_WORDS",
            "REDUNDANCY",
            "TYPOS",
            "STYLE",
            "PROPER_NOUNS",
            "IDIOMS",
            "DE_DOUBLE_PUNCTUATION",
            "DOUBLE_EXCLAMATION_MARK",
        ]
        return names

    def __call__(self, text: str) -> np.array:
        mistakes = self.spell_checker.check(text)

        output = np.zeros((1, self.dim))

        if len(mistakes) > 0:
            for mistake in mistakes:
                if mistake.category == "MISC":
                    output[0, 0] += 1
                elif mistake.category == "EMPFOHLENE_RECHTSCHREIBUNG":
                    output[0, 1] += 1
                elif mistake.category == "TYPOGRAPHY" and mistake.ruleId not in [
                    "TYPOGRAFISCHE_ANFUEHRUNGSZEICHEN",
                    "FALSCHE_VERWENDUNG_DES_BINDESTRICHS",
                    "AKZENT_STATT_APOSTROPH",
                    "MULTIPLICATION_SIGN",
                ]:
                    output[0, 2] += 1
                elif mistake.category == "PUNCTUATION" and mistake.ruleId not in [
                    "EINHEIT_LEERZEICHEN",
                    "ZEICHENSETZUNG_DIREKTE_REDE",
                ]:
                    output[0, 3] += 1
                elif mistake.category == "GRAMMAR":
                    output[0, 4] += 1
                elif mistake.category == "CASING":
                    output[0, 5] += 1
                elif mistake.category == "HILFESTELLUNG_KOMMASETZUNG":
                    output[0, 6] += 1
                elif mistake.category == "COLLOQUIALISMS":
                    output[0, 7] += 1
                elif mistake.category == "COMPOUNDING":
                    output[0, 8] += 1
                elif mistake.category == "CONFUSED_WORDS":
                    output[0, 9] += 1
                elif mistake.category == "REDUNDANCY":
                    output[0, 10] += 1
                elif mistake.category == "TYPOS":
                    output[0, 11] += 1
                elif mistake.category == "STYLE":
                    output[0, 12] += 1
                elif mistake.category == "PROPER_NOUNS":
                    output[0, 13] += 1
                elif mistake.category == "IDIOMS":
                    output[0, 14] += 1
                elif mistake.category == "PUNCTUATION" and mistake.ruleId == "DE_DOUBLE_PUNCTUATION":
                    output[0, 15] += 1
                elif mistake.category == "PUNCTUATION" and mistake.ruleId == "DOPPELTES_AUSRUFEZEICHEN":
                    output[0, 16] += 1

        return np.log(output + 1e-9) - np.log(len(text.split()) + 1e-9)


class DocumentEmbeddingsFastTextPool(Feature):
    """
    A feature that computes document embeddings using FastText pre-trained word embeddings and pooling.

    This class utilizes FastText word embeddings for encoding text into document-level embeddings. The document
    embeddings are derived by pooling word embeddings obtained from the FastText model.

    Attributes
    ----------
    document_embeddings : DocumentPoolEmbeddings
        An instance of `DocumentPoolEmbeddings` that pools FastText word embeddings to generate document embeddings.

    Methods
    -------
    __init__() -> None
        Initializes the `DocumentEmbeddingsFastTextPool` feature with FastText word embeddings.
    dim() -> int
        Returns the dimensionality of the embeddings, which is 300.
    type() -> str
        Returns the type of the feature, which is "embedding".
    is_trainable() -> bool
        Indicates whether the feature is trainable. Always returns False.
    __call__(text: str) -> np.array
        Computes the document embedding for the given text using FastText embeddings.
    """

    def __init__(self) -> None:
        word_embeddings = WordEmbeddings("de")
        self.document_embeddings = DocumentPoolEmbeddings([word_embeddings])

    @property
    def dim(self):
        return 300

    @property
    def type(self):
        return "embedding"

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        sentence = Sentence(text)
        self.document_embeddings.embed(sentence)

        return sentence.embedding.detach().cpu().numpy()


class DocumentEmbeddingsBERT(Feature):
    """
    A feature that computes document embeddings using BERT pre-trained embeddings.

    This class utilizes a BERT model ("bert-base-german-cased") to encode text into document-level embeddings. The
    embeddings are computed using the `TransformerDocumentEmbeddings` class, which provides embeddings from a BERT model
    and does not perform fine-tuning.

    Attributes
    ----------
    embeddings : TransformerDocumentEmbeddings
        An instance of `TransformerDocumentEmbeddings` configured with BERT for German language.

    Methods
    -------
    __init__() -> None
        Initializes the `DocumentEmbeddingsBERT` feature with the BERT model.
    dim() -> int
        Returns the dimensionality of the embeddings, which is 768.
    type() -> str
        Returns the type of the feature, which is "embedding".
    is_trainable() -> bool
        Indicates whether the feature is trainable. Always returns False.
    __call__(text: str) -> np.array
        Computes the document embedding for the given text using the BERT model.
    """

    def __init__(self) -> None:
        self.embeddings = TransformerDocumentEmbeddings("bert-base-german-cased", fine_tune=False)

    @property
    def dim(self):
        return 768

    @property
    def type(self):
        return "embedding"

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        sentence = Sentence(text)
        self.embeddings.embed(sentence)

        return sentence.embedding.detach().cpu().numpy()


class SentimentTextBlob(Feature):
    """
    A feature that computes the sentiment polarity of a given text using TextBlob.

    This class uses the TextBlob library to evaluate the sentiment of the text. The sentiment polarity score ranges from
     -1 (very negative) to 1 (very positive), indicating the overall sentiment of the text.

    Methods
    -------
    dim() -> int
        Returns the dimensionality of the feature, which is 1.
    type() -> str
        Returns the type of the feature, which is "numerical".
    is_trainable() -> bool
        Indicates whether the feature is trainable. Always returns False.
    __call__(text: str) -> float
        Computes the sentiment polarity score for the given text using TextBlob.
    """

    @property
    def dim(self):
        return 1

    @property
    def type(self):
        return "numerical"

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> float:
        blob = TextBlob(text)

        return blob.sentiment.polarity


class SentimentBERT(Feature):
    """
    A feature that computes sentiment scores using a BERT-based model for sequence classification.

    This class uses a pre-trained BERT model specifically fine-tuned for sentiment analysis in German. It processes
    the input text, extracts sentiment scores, and provides them as numerical features. The model outputs three
    sentiment scores corresponding to different sentiment classes.

    Attributes
    ----------
    MAX_NUM_TOKENS : int
        The maximum number of tokens for input sequences, set to 512. Sequences longer than this are truncated.
    device : str
        The device on which the model is loaded, either "cuda" (GPU) if available or "cpu".
    model : transformers.AutoModelForSequenceClassification
        The pre-trained BERT model for sequence classification.
    tokenizer : transformers.AutoTokenizer
        The tokenizer associated with the pre-trained BERT model.
    clean_chars : re.Pattern
        A regular expression pattern for cleaning non-alphabetic characters from the text.
    clean_http_urls : re.Pattern
        A regular expression pattern for removing URLs from the text.
    clean_at_mentions : re.Pattern
        A regular expression pattern for removing "@mentions" from the text.

    Methods
    -------
    __init__() -> None
        Initializes the SentimentBERT feature with a pre-trained BERT model and tokenizer.
    __call__(text: str) -> np.array
        Computes the sentiment scores for the given text using the BERT model.
    _clean_text(text: str) -> str
        Cleans the input text by removing URLs, mentions, non-alphabetic characters, and normalizing whitespace.
    _replace_numbers(text: str) -> str
        Replaces numerical digits in the text with their German word equivalents.
    """

    MAX_NUM_TOKENS = 512

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForSequenceClassification.from_pretrained("oliverguhr/german-sentiment-bert")
        self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained("oliverguhr/german-sentiment-bert")
        self.clean_chars = re.compile(r"[^A-Za-züöäÖÜÄß ]", re.MULTILINE)
        self.clean_http_urls = re.compile(r"https*\S+", re.MULTILINE)
        self.clean_at_mentions = re.compile(r"@\S+", re.MULTILINE)

    @property
    def dim(self):
        return 3

    @property
    def type(self):
        return "numerical"

    @property
    def is_trainable(self) -> bool:
        return False

    def __call__(self, text: str) -> np.array:
        text = self._clean_text(text)

        input_ids = self.tokenizer.batch_encode_plus([text], padding=True, add_special_tokens=True)
        input_ids = torch.tensor(input_ids["input_ids"])
        input_ids = input_ids.to(self.device)

        if input_ids.shape[-1] > self.MAX_NUM_TOKENS:
            input_ids = input_ids[:, : self.MAX_NUM_TOKENS]

        with torch.no_grad():
            result = self.model(input_ids)

        return result.logits.squeeze().detach().cpu().numpy()

    def _clean_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = self.clean_http_urls.sub("", text)
        text = self.clean_at_mentions.sub("", text)
        text = self._replace_numbers(text)
        text = self.clean_chars.sub("", text)
        text = " ".join(text.split())
        text = text.strip().lower()

        return text

    @staticmethod
    def _replace_numbers(text: str) -> str:
        return (
            text.replace("0", " null")
            .replace("1", " eins")
            .replace("2", " zwei")
            .replace("3", " drei")
            .replace("4", " vier")
            .replace("5", " fünf")
            .replace("6", " sechs")
            .replace("7", " sieben")
            .replace("8", " acht")
            .replace("9", " neun")
        )


class WritingStyleEmbeddings(Feature):
    """
    A feature that generates writing style embeddings using a pre-trained AdHominem model.

    This class uses the AdHominem model to extract writing style embeddings from text. It preprocesses the input text
    using the SoMaJo tokenizer and then computes embeddings via the AdHominem model. The model parameters are loaded
    from a specified file, and TensorFlow is used to handle the model's computation.

    Attributes
    ----------
    tokenizer : SoMaJo
        The tokenizer used to preprocess the text, configured for the specified language.
    adhominem : AdHominem
        The AdHominem model for generating embeddings, initialized with pre-loaded parameters.
    session : tf.Session
        The TensorFlow session used to run the model computations.

    Methods
    -------
    __init__() -> None
        Initializes the WritingStyleEmbeddings feature by setting up the tokenizer, loading the AdHominem model,
        and initializing the TensorFlow session.
    __call__(text: str) -> np.array
        Computes the writing style embeddings for the given text using the AdHominem model.
    preprocess_doc(doc: str, tokenizer: SoMaJo) -> str
        Preprocesses the input text by tokenizing it into sentences and concatenating tokens.
    make_inference(doc: str, adhominem: AdHominem, sess: tf.Session) -> np.array
        Performs inference using the AdHominem model to generate embeddings for the given preprocessed text.
    """

    def __init__(self) -> None:
        self.tokenizer = SoMaJo(language="en_PTB", split_sentences=True)  # "de_CMC"

        tf.reset_default_graph()

        with open(os.path.join("data", "adhominem.skops"), "rb") as f:
            parameters = load(f)  # type: ignore

        with tf.variable_scope("AdHominem"):
            self.adhominem = AdHominem(
                hyper_parameters=parameters["hyper_parameters"],
                theta_init=parameters["theta"],
                theta_E_init=parameters["theta_E"],
            )

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    @property
    def dim(self):
        return 100

    @property
    def type(self):
        return "embedding"

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
        doc_new = ""
        for sentence in sentences:
            for token in sentence:
                doc_new += token.text + " "
        doc_new = doc_new.strip()

        return doc_new

    @staticmethod
    def make_inference(doc, adhominem, sess):
        emb = adhominem.inference([doc], sess)

        return emb


class FeatureExtractor:
    """
    A class for extracting features from a dataset using specified feature functions.

    The `FeatureExtractor` class applies a list of `Feature` functions to a dataset, computing features and optionally
    saving them to a file. It handles both training and inference stages, allowing for feature extraction from text data
    and label management.

    Parameters
    ----------
    feature_funcs : List[Feature]
        A list of `Feature` objects to be used for feature extraction. Each `Feature` object should implement the
        `__call__` method to process text and return feature values.

    Methods
    -------
    get_features(
        dataset_file: Union[str, Path],
        save_file: Optional[Union[str, Path]] = None,
        train: bool = False,
        show_progress_bar: bool = False,
    ) -> Tuple[np.array, np.array]
        Extracts features from the dataset specified by `dataset_file`. Optionally saves the features and labels to
        `save_file`. If `train` is True, fits the trainable features on the training data. Optionally displays a
        progress bar.

    _check_has_labels(data_frame: pd.DataFrame) -> bool
        Checks if the dataset contains label columns necessary for training or evaluation.

    _get_feature_dim() -> int
        Computes the total dimensionality of the feature space based on the dimensions of the provided feature functions.
    """

    def __init__(self, feature_funcs: List[Feature]) -> None:
        self.feature_funcs = feature_funcs

    def get_features(
        self,
        dataset_file: Union[str, Path],
        save_file: Optional[Union[str, Path]] = None,
        train: bool = False,
        show_progress_bar: bool = False,
    ) -> Tuple[np.array, np.array]:
        if save_file is not None and os.path.isfile(save_file):
            file = np.load(save_file, allow_pickle=True)
            features, labels = file["features"], file["labels"]
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

            for row in tqdm(
                data_frame.iterrows(), desc="Computing features", total=len(data_frame), disable=not show_progress_bar
            ):
                row_idx, text = row[0], row[1][1]

                feature_idx = 0
                for feature_func in self.feature_funcs:
                    features[row_idx, feature_idx : feature_idx + feature_func.dim] = feature_func(text)
                    feature_idx += feature_func.dim

                if has_labels:
                    labels[row_idx, :] = row[1][2:].to_numpy()

            if save_file is not None:
                np.savez(save_file, features=features, labels=labels)

        return features, labels

    @staticmethod
    def _check_has_labels(data_frame: pd.DataFrame) -> bool:
        return (
            "Sub1_Toxic" in data_frame.columns
            and "Sub2_Engaging" in data_frame.columns
            and "Sub3_FactClaiming" in data_frame.columns
        )

    def _get_feature_dim(self):
        feature_dim = 0

        for feature_func in self.feature_funcs:
            feature_dim += feature_func.dim

        return feature_dim
