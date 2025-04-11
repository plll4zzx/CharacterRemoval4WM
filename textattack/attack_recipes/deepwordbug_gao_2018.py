"""

DeepWordBug
========================================
(Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers)

"""

from textattack import Attack
from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification,InputReduction
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import (
    CompositeTransformation,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
)

from .attack_recipe import AttackRecipe


class DeepWordBugGao2018(AttackRecipe):
    """Gao, Lanchantin, Soffa, Qi.

    Black-box Generation of Adversarial Text Sequences to Evade Deep
    Learning Classifiers.

    https://arxiv.org/abs/1801.04354
    """

    @staticmethod
    def build(model_wrapper, use_all_transformations=True, query_budget=100, edit_distance=20):
        #
        # Swap characters out from words. Choose the best of four potential transformations.
        #
        if use_all_transformations:
            # We propose four similar methods:
            transformation = CompositeTransformation(
                [
                    # (1) Swap: Swap two adjacent letters in the word.
                    WordSwapNeighboringCharacterSwap(),
                    # (2) Substitution: Substitute a letter in the word with a random letter.
                    WordSwapRandomCharacterSubstitution(),
                    # (3) Deletion: Delete a random letter from the word.
                    WordSwapRandomCharacterDeletion(),
                    # (4) Insertion: Insert a random letter in the word.
                    WordSwapRandomCharacterInsertion(),
                ],
                max_trans_num=20
            )
        else:
            # We use the Combined Score and the Substitution Transformer to generate
            # adversarial samples, with the maximum edit distance difference of 30
            # (ϵ = 30).
            transformation = WordSwapRandomCharacterSubstitution()
        #
        # Don't modify the same word twice or stopwords
        #
        constraints = [RepeatModification(), StopwordModification()]
        #
        # In these experiments, we hold the maximum difference
        # on edit distance (ϵ) to a constant 30 for each sample.
        #
        constraints.append(LevenshteinEditDistance(edit_distance))
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper, query_budget=query_budget)
        #
        # Greedily swap words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR(max_index_num=20)

        return Attack(goal_function, constraints, transformation, search_method)
