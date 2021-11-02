from argparse import ArgumentParser
from collections import defaultdict
from typing import DefaultDict, Optional

import numpy as np
from hunspell import Hunspell
from metaphone import doublemetaphone
from onnxruntime import InferenceSession
from textdistance import needleman_wunsch, editex, jaro_winkler

from base_spell_cheker import BaseSpellChecker, main


class HunSpellChecker(BaseSpellChecker):
    def __init__(self,
                 model=None,
                 session: Optional[InferenceSession] = None,
                 word_frequency_file_path: str = 'data/train/unigram_freq.csv'
                 ) -> None:
        super().__init__(model, session, 4)

        self.dictionary = Hunspell('en_US')
        self.word_probabilities = self.__get_word_probabilities(word_frequency_file_path)

    @staticmethod
    def __get_word_probabilities(word_frequency_file_path: str) -> DefaultDict[str, float]:
        words = []
        word_frequency = []
        total_frequency = 0
        with open(word_frequency_file_path, "r") as f:
            for line in f:
                word, frequency = line.lower().strip().split(",")
                frequency = int(frequency)
                words.append(word)
                word_frequency.append(frequency)
                total_frequency += frequency
        for word_id in range(len(word_frequency)):
            word_frequency[word_id] /= total_frequency

        return defaultdict(int, zip(words, word_frequency))

    def _get_suggested_words(self, word: str, max_suggestions: int = None):
        return np.array(self.dictionary.suggest(word))[:max_suggestions]

    def _get_features(self, word: str, suggested_word: str) -> np.ndarray:
        phonetic_world = doublemetaphone(word)[0]
        phonetic_suggested_word = doublemetaphone(suggested_word)[0]
        phonetic_distance = editex.normalized_distance(phonetic_world, phonetic_suggested_word)

        keyboard_distance = needleman_wunsch.normalized_distance(word, suggested_word)

        probability_suggested_word = self.word_probabilities[suggested_word]

        jaro_winkler_distance = jaro_winkler.normalized_distance(word, suggested_word)

        return np.array([
            phonetic_distance,  # Фонетические расстояния
            keyboard_distance,  # Вероятность опечатки по клавиатуре
            1 - probability_suggested_word,  # Корпусная вероятность слова
            jaro_winkler_distance  # Jaro-Winkler
        ])


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--word", type=str, required=True)
    args.add_argument("--max_suggestions", type=int, default=10)

    args = args.parse_args()

    main(HunSpellChecker(), args.word, args.max_suggestions)
