from argparse import ArgumentParser
from collections import Counter
from collections import defaultdict
from typing import List
from typing import Optional

import numpy as np
from metaphone import doublemetaphone
from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from spylls.hunspell import Dictionary
from textdistance import needleman_wunsch, editex, jaro_winkler
from tqdm.auto import tqdm


class SpellChecker(object):
    def __init__(self,
                 model=None,
                 session: Optional[InferenceSession] = None,
                 ) -> None:
        dictionary = Dictionary.from_files("en_US")
        self.dictionary = np.unique([word.stem.lower() for word in dictionary.dic.words])

        self.model = model
        self.session = session

        self.vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(2, 2), binary=True)
        encoded_words = self.vectorizer.fit_transform(self.dictionary).tocoo()

        self.index = defaultdict(set)

        # index[ngram] = [word1, word2, ...]
        for i in zip(encoded_words.row, encoded_words.col):
            self.index[i[1]].add(i[0])

    def __get_suggested_words(self, word: str, max_suggestions: int = None):
        char_ngrams_list = self.vectorizer.transform([word]).tocoo().col
        counter = Counter()
        for token_id in char_ngrams_list:
            for word_id in self.index[token_id]:
                counter[word_id] += 1

        return np.array(
            [self.dictionary[suggested_word[0]] for suggested_word in counter.most_common(2 * max_suggestions)]
        )

    def __call__(self, word: str, max_suggestions: int = 10) -> np.ndarray:
        suggested_words = self.__get_suggested_words(word, 2 * max_suggestions)

        if len(suggested_words) == 0:
            return np.array([])

        features = np.vstack([self.__get_features(word, suggested_word) for suggested_word in suggested_words])

        if self.session is None:
            return self.__ranking_suggested_words(suggested_words, features, max_suggestions)
        return self.__onnx_ranking_suggested_words(suggested_words, features, max_suggestions)

    def __get_features(self, word: str, suggested_word: str) -> List[float]:
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
        initial_type = [('float_input', FloatTensorType([None, 3]))]
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
    spell_checker = SpellChecker()
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