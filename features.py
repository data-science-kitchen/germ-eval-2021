import numpy as np
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


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
