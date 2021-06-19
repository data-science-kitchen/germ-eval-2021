from flair.tokenization import Token, Tokenizer
from typing import List


class SpaceAndURLTokenizer(Tokenizer):
    """
        Tokenizer based on space and slash (for URLs) character only.
    """
    def __init__(self):
        super(SpaceAndURLTokenizer, self).__init__()

    def tokenize(self, text: str) -> List[Token]:
        return SpaceAndURLTokenizer.run_tokenize(text)

    @staticmethod
    def run_tokenize(text: str) -> List[Token]:
        tokens: List[Token] = []
        word = ""
        index = -1
        for index, char in enumerate(text):
            if char == " " or char == "/" or char == "-" or char == "." or char == ":":
                if len(word) > 0:
                    start_position = index - len(word)
                    tokens.append(
                        Token(
                            text=word, start_position=start_position, whitespace_after=True
                        )
                    )

                word = ""
            else:
                word += char
        # increment for last token in sentence if not followed by whitespace
        index += 1
        if len(word) > 0:
            start_position = index - len(word)
            tokens.append(
                Token(text=word, start_position=start_position, whitespace_after=False)
            )

        return tokens
