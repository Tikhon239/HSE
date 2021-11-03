from typing import Optional

import numpy as np
from onnxruntime import InferenceSession
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm


class BaseSpellChecker:
    def __init__(self,
                 model=None,
                 session: Optional[InferenceSession] = None,
                 number_of_features: int = None
                 ) -> None:
        self.model = model
        self.session = session
        self.number_of_features = number_of_features

    def _get_suggested_words(self, word: str, max_suggestions: int = None) -> np.ndarray:
        raise NotImplementedError()

    def _get_features(self, word: str, suggested_word: str) -> np.ndarray:
        raise NotImplementedError()

    def __call__(self, word: str, max_suggestions: int = 10) -> np.ndarray:
        suggested_words = self._get_suggested_words(word, 2 * max_suggestions)

        if len(suggested_words) == 0:
            return np.array([])

        features = np.vstack([self._get_features(word, suggested_word) for suggested_word in suggested_words])

        if self.session is None:
            return self.__ranking_suggested_words(suggested_words, features, max_suggestions)
        return self.__onnx_ranking_suggested_words(suggested_words, features, max_suggestions)

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
                for wrong_word in self._get_suggested_words(word, 2):
                    if wrong_word != correction_word:
                        wrong_features.append(self._get_features(word, wrong_word))
                        correction_features.append(self._get_features(word, correction_word))
                        break

        features = np.vstack((wrong_features, correction_features))
        target = np.zeros(len(features))
        target[:len(wrong_features)] = 1
        self.model.fit(features, target)

    def convert_model(self, output_path: str) -> None:
        initial_type = [('float_input', FloatTensorType([None, self.number_of_features]))]
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


def main(spell_checker: BaseSpellChecker, word: str, max_suggestions: int = 10) -> None:
    suggested_words = spell_checker(word, max_suggestions)
    visualizate_suggested_words(word, suggested_words)

    print("Add logistic regression")
    spell_checker.set_model(LogisticRegression(n_jobs=-1))
    spell_checker.fit_model('data/train/train.tsv')
    suggested_words = spell_checker(word, max_suggestions)
    visualizate_suggested_words(word, suggested_words)

    print("Check onnx run")
    spell_checker.convert_model('data/debug/model.onnx')
    session = InferenceSession('data/debug/model.onnx')
    spell_checker.set_session(session)

    onnx_suggested_words = spell_checker(word, max_suggestions)
    visualizate_suggested_words(word, onnx_suggested_words)