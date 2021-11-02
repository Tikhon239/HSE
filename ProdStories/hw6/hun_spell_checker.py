from argparse import ArgumentParser
from collections import defaultdict
from typing import DefaultDict, Optional

import numpy as np
from hunspell import Hunspell
from metaphone import doublemetaphone
from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
from textdistance import needleman_wunsch, editex, jaro_winkler
from tqdm.auto import tqdm


class HunSpellChecker:
    def __init__(self,
                 model=None,
                 session: Optional[InferenceSession] = None,
                 word_frequency_file_path: str = 'data/train/unigram_freq.csv'
                 ) -> None:
        self.dictionary = Hunspell('en_US')
        self.model = model
        self.session = session
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

    def __get_suggested_words(self, word: str, max_suggestions: int = None):
        return np.array(self.dictionary.suggest(word))[:max_suggestions]

    def __call__(self, word: str, max_suggestions: int = 10) -> np.ndarray:
        if self.dictionary.spell(word):
            return np.array([word])

        suggested_words = self.__get_suggested_words(word)

        if len(suggested_words) == 0:
            return np.array([])

        features = np.vstack([self.__get_features(word, suggested_word) for suggested_word in suggested_words])

        if self.session is None:
            return self.__ranking_suggested_words(suggested_words, features, max_suggestions)
        return self.__onnx_ranking_suggested_words(suggested_words, features, max_suggestions)

    def __get_features(self, word: str, suggested_word: str) -> np.ndarray:
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

    def __ranking_suggested_words(self,
                                  suggested_words: np.ndarray,
                                  features: np.ndarray,
                                  max_suggestions: int = 10
                                  ) -> np.ndarray:
        if self.model is None:
            generalized_features = np.mean(features, axis=1)
        else:
            generalized_features = self.model.predict_proba(features)[:, 1]

        return suggested_words[np.argsort(generalized_features)[:max_suggestions]]

    def __onnx_ranking_suggested_words(self,
                                       suggested_words: np.ndarray,
                                       features: np.ndarray,
                                       max_suggestions: int = 10
                                       ) -> np.ndarray:
        input_name = self.session.get_inputs()[0].name
        label_name = self.session.get_outputs()[1].name
        generalized_features = self.session.run([label_name], {input_name: features.astype(np.float32)})[0]
        generalized_features = [generalized_feature[1] for generalized_feature in generalized_features]
        return suggested_words[np.argsort(generalized_features)[:max_suggestions]]

    def fit_model(self, file_path: str = 'data/train/train.tsv') -> None:
        wrong_features = []
        correction_features = []
        with open(file_path, "r") as f:
            for i, line in enumerate(tqdm(f, desc='Build dataset')):
                if i % 10 != 0:  # слишком много данных
                    continue
                word, correction_word = line.lower().strip().split("\t")
                for wrong_word in self.__get_suggested_words(word, 2):
                    if wrong_word != correction_word:
                        wrong_features.append(self.__get_features(word, wrong_word))
                        correction_features.append(self.__get_features(word, correction_word))
                        break

        features = np.vstack((wrong_features, correction_features))
        target = np.ones(len(features))
        target[:len(wrong_features)] = 0
        self.model.fit(features, target)

    def convert_model(self, output_path: str) -> None:
        initial_type = [('float_input', FloatTensorType([None, 4]))]
        onx = convert_sklearn(self.model, initial_types=initial_type)
        with open(output_path, 'wb') as f:
            f.write(onx.SerializeToString())

    def set_model(self, model) -> None:
        self.model = model

    def set_session(self, session: InferenceSession) -> None:
        self.session = session


def visualizate_suggested_words(word: str, suggested_words: np.ndarray) -> None:
    print(f"Word: {word}")
    print(f"Suggested words: ")

    for pos, suggested_word in enumerate(suggested_words):
        print(f"{pos + 1:{' '}{'<'}{3}}{suggested_word}")


def main(word: str, max_suggestions: int = 10) -> None:
    spell_checker = HunSpellChecker()
    suggested_words = spell_checker(word, max_suggestions)
    visualizate_suggested_words(word, suggested_words)

    spell_checker.set_model(LogisticRegression(n_jobs=-1))
    spell_checker.fit_model('data/train/train.tsv')
    suggested_words = spell_checker(word, max_suggestions)
    visualizate_suggested_words(word, suggested_words)

    spell_checker.convert_model('data/debug/model.onnx')
    session = InferenceSession('data/debug/model.onnx')
    spell_checker.set_session(session)

    onnx_suggested_words = spell_checker(word, max_suggestions)
    visualizate_suggested_words(word, onnx_suggested_words)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--word", type=str, required=True)
    args.add_argument("--max_suggestions", type=int, default=10)

    args = args.parse_args()

    main(args.word, args.max_suggestions)
