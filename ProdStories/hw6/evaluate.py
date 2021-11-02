from typing import List, Union, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from hun_spell_checker import HunSpellChecker
from my_spell_checker import SpellChecker


def get_data(file_path: str) -> Tuple[List[str], List[str]]:
    wrong_words = []
    correction_words = []
    with open(file_path, "r") as f:
        for line in f:
            wrong_word, correction_word = line.lower().strip().split("\t")
            wrong_words.append(wrong_word)
            correction_words.append(correction_word)

    return wrong_words, correction_words


def get_accuracy(suggested_words_array: List[np.ndarray], correction_words: List[str]) -> float:
    total_correction = 0
    for suggested_words, correction_word in zip(suggested_words_array, correction_words):
        total_correction += correction_word in suggested_words
    return total_correction / len(correction_words)


def evaluate(file_path: str, spell_checker: Union[HunSpellChecker, SpellChecker], max_suggestions: int = 10) -> None:
    wrong_words, correction_words = get_data(file_path)
    suggested_words_array = []

    for wrong_word in tqdm(wrong_words, desc="Predicting"):
        suggested_words_array.append(spell_checker(wrong_word, max_suggestions))

    print(f"Acc@{max_suggestions}: {get_accuracy(suggested_words_array, correction_words): .3f}")


if __name__ == "__main__":
    file_path = 'data/test/test_data.txt'
    spell_checker = SpellChecker()
    simple_hun_spell_checker = HunSpellChecker()
    hun_spell_checker = HunSpellChecker(model=LogisticRegression(n_jobs=-1))
    hun_spell_checker.fit_model('data/train/train.tsv')

    for spell_checker in [spell_checker, simple_hun_spell_checker, hun_spell_checker]:
        print(spell_checker.__class__.__name__)
        for max_suggestions in [1, 5, 10]:
            evaluate(file_path, spell_checker, max_suggestions)
