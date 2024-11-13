"""

SemanticGoalFunctionResult Class
====================================

text2text goal function Result

"""

import torch

import textattack
from textattack.shared import utils

from .goal_function_result import GoalFunctionResult


class SemanticGoalFunctionResult(GoalFunctionResult):
    """Represents the result of a text-to-text goal function."""

    def __init__(
        self,
        attacked_text,
        # raw_output,
        # output,
        goal_status,
        score,
        num_queries,
        ground_truth_output,
    ):
        super().__init__(
            attacked_text,
            None,#raw_output,
            None,#output,
            goal_status,
            score,
            num_queries,
            ground_truth_output,
            goal_function_result_type="Semantic",
        )

    # @property
    # def _processed_output(self):
    #     """Takes a model output (like `1`) and returns the class labeled output
    #     (like `positive`), if possible.

    #     Also returns the associated color.
    #     """
    #     output_label = 1-self.score
    #     if 
    #     if self.attacked_text.attack_attrs.get("label_names") is not None:
    #         output = self.attacked_text.attack_attrs["label_names"][self.output]
    #         output = textattack.shared.utils.process_label_name(output)
    #         color = textattack.shared.utils.color_from_output(output, output_label)
    #         return output, color
    #     else:
    #         color = textattack.shared.utils.color_from_label(output_label)
    #         return output_label, color

    def get_text_color_input(self):
        """A string representing the color this result's changed portion should
        be if it represents the original input."""
        return "red"

    def get_text_color_perturbed(self):
        """A string representing the color this result's changed portion should
        be if it represents the perturbed input."""
        return "blue"

    def get_colored_output(self, color_method=None):
        """Returns a string representation of this result's output, colored
        according to `color_method`."""
        return str(self.output)
