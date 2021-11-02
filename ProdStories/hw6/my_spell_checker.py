from argparse import ArgumentParser
from collections import Counter
from collections import defaultdict
from typing import List
from typing import Optional

import numpy as np
from metaphone import doublemetaphone
from onnxruntime import InferenceSession
from sklearn.feature_extraction.text import CountVectorizer
from spylls.hunspell import Dictionary
from textdistance import needleman_wunsch, editex, jaro_winkler

from base_spell_cheker import BaseSpellChecker, main


class SpellChecker(BaseSpellChecker):
    def __init__(self,
                 model=None,
                 session: Optional[InferenceSession] = None,
                 ) -> None:
        super().__init__(model, session, 3)

        dictionary = Dictionary.from_files("en_US")
        self.dictionary = np.unique([word.stem.lower() for word in dictionary.dic.words])

        self.vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(2, 2), binary=True)
        encoded_words = self.vectorizer.fit_transform(self.dictionary).tocoo()

        self.index = defaultdict(set)

        # index[ngram] = [word1, word2, ...]
        for i in zip(encoded_words.row, encoded_words.col):
            self.index[i[1]].add(i[0])

    def _get_suggested_words(self, word: str, max_suggestions: int = None):
        char_ngrams_list = self.vectorizer.transform([word]).tocoo().col
        counter = Counter()
        for token_id in char_ngrams_list:
            for word_id in self.index[token_id]:
                counter[word_id] += 1

        return np.array(
            [self.dictionary[suggested_word[0]] for suggested_word in counter.most_common(max_suggestions)]
        )

    def _get_features(self, word: str, suggested_word: str) -> List[float]:
        phonetic_world = doublemetaphone(word)[0]
        phonetic_suggested_word = doublemetaphone(suggested_word)[0]
        phonetic_distance = editex.normalized_distance(phonetic_world, phonetic_suggested_word)

        keyboard_distance = needleman_wunsch.normalized_distance(word, suggested_word)

        jaro_winkler_distance = jaro_winkler.normalized_distance(word, suggested_word)

        return [
            phonetic_distance,  # Фонетические расстояния
            keyboard_distance,  # Вероятность опечатки по клавиатуре
            jaro_winkler_distance  # Jaro-Winkler
        ]


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--word", type=str, required=True)
    args.add_argument("--max_suggestions", type=int, default=10)

    args = args.parse_args()

    main(SpellChecker(), args.word, args.max_suggestions)