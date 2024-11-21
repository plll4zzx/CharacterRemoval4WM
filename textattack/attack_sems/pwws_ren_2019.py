"""

PWWS
=======

(Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency)

"""

from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapWordNet
from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.goal_functions import UntargetedClassification, SemanticGoalFunction, UntargetedSemantic

from .attack_sem import AttackSem


class PWWSRen2019(AttackSem):
    """An implementation of Probability Weighted Word Saliency from "Generating
    Natural Language Adversarial Examples through Probability Weighted Word
    Saliency", Ren et al., 2019.

    Words are prioritized for a synonym-swap transformation based on a
    combination of their saliency score and maximum word-swap
    effectiveness. Note that this implementation does not include the
    Named Entity adversarial swap from the original paper, because it
    requires access to the full dataset and ground truth labels in
    advance.

    https://www.aclweb.org/anthology/P19-1103/
    """

    @staticmethod
    def build(model_wrapper, target_cos=0.7, edit_distance=10, query_budget=100):
        transformation = WordSwapWordNet()
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(LevenshteinEditDistance(edit_distance))
        goal_function = UntargetedSemantic(model_wrapper, target_cos=target_cos, query_budget=query_budget)
        # search over words based on a combination of their saliency score, and how efficient the WordSwap transform is
        search_method = GreedyWordSwapWIR("weighted-saliency")
        return Attack(goal_function, constraints, transformation, search_method)
