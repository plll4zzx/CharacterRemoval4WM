"""

TextBugger
===============

(TextBugger: Generating Adversarial Text Against Real-world Applications)

"""

from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.goal_functions import UntargetedClassification, UntargetedSemantic
from textattack.search_methods import GreedyWordSwapWIR, SampleWordSwapWIR, SlidingWindowWSample
from textattack.transformations import (
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
)

from .attack_sem import AttackSem
import string

class TextBuggerLi2018(AttackSem):
    """Li, J., Ji, S., Du, T., Li, B., and Wang, T. (2018).

    TextBugger: Generating Adversarial Text Against Real-world Applications.

    https://arxiv.org/abs/1812.05271
    """

    @staticmethod
    def build(
        model_wrapper, 
        target_cos=0.7, edit_distance=10, query_budget=100, 
        random_num=5, random_one=True, 
        temperature=30, max_single_query=20, slide_flag=True,
        window_size=3
    ):
        #
        #  we propose five bug generation methods for TEXTBUGGER:
        #
        # unprintable_char=''.join([chr(i) for i in range(1000) if chr(i).isprintable()==False])[0:10]
        # special_char=''.join([chr(i) for i in range(0,500,10) if chr(i).isprintable() and chr(i) not in string.printable])[0:10]
        transformation = CompositeTransformation(
            [
                # (1) Insert: Insert a space into the word.
                # Generally, words are segmented by spaces in English. Therefore,
                # we can deceive classifiers by inserting spaces into words.
                WordSwapRandomCharacterInsertion(
                    random_one=random_one,
                    # letters_to_insert=string.punctuation,#+string.octdigits+string.whitespace,
                    # letters_to_insert=unprintable_char+special_char+string.whitespace,
                    letters_to_insert=string.whitespace,#string.printable,#
                    skip_first_char=True,
                    skip_last_char=True,
                    random_num=random_num
                ),
                # (2) Delete: Delete a random character of the word except for the first
                # and the last character.
                WordSwapRandomCharacterDeletion(
                    random_one=random_one, skip_first_char=True, skip_last_char=True,
                    random_num=random_num,
                ),
                # (3) Swap: Swap random two adjacent letters in the word but do not
                # alter the first or last letter. This is a common occurrence when
                # typing quickly and is easy to implement.
                WordSwapNeighboringCharacterSwap(
                    random_one=random_one, skip_first_char=True, skip_last_char=True,
                    random_num=random_num,
                ),
                # (4) Substitute-C (Sub-C): Replace characters with visually similar
                # characters (e.g., replacing “o” with “0”, “l” with “1”, “a” with “@”)
                # or adjacent characters in the keyboard (e.g., replacing “m” with “n”).
                WordSwapHomoglyphSwap(),
                # (5) Substitute-W
                # (Sub-W): Replace a word with its topk nearest neighbors in a
                # context-aware word vector space. Specifically, we use the pre-trained
                # GloVe model [30] provided by Stanford for word embedding and set
                # topk = 5 in the experiment.
                WordSwapEmbedding(max_candidates=5),
            ]
        )

        constraints = [RepeatModification(), StopwordModification()]
        # In our experiment, we first use the Universal Sentence
        # Encoder [7], a model trained on a number of natural language
        # prediction tasks that require modeling the meaning of word
        # sequences, to encode sentences into high dimensional vectors.
        # Then, we use the cosine similarity to measure the semantic
        # similarity between original texts and adversarial texts.
        # ... "Furthermore, the semantic similarity threshold \eps is set
        # as 0.8 to guarantee a good trade-off between quality and
        # strength of the generated adversarial text."
        # constraints.append(UniversalSentenceEncoder(threshold=0.8))
        constraints.append(LevenshteinEditDistance(edit_distance))
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedSemantic(model_wrapper, target_cos=target_cos, query_budget=query_budget, max_single_query=max_single_query)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        if slide_flag:
            search_method = SlidingWindowWSample(wir_method="delete", temperature=temperature, window_size=window_size, step_size=1)
        else:
            search_method = SampleWordSwapWIR(wir_method="delete", temperature=temperature)

        return Attack(goal_function, constraints, transformation, search_method)
