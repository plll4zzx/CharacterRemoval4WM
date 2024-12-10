"""

Goal Function for Semantic
-------------------------------------------------------
"""
import numpy as np
import torch

from textattack.goal_function_results import SemanticGoalFunctionResult
from textattack.goal_functions import GoalFunction


class SemanticGoalFunction(GoalFunction):
    """A goal function defined on a model that outputs a probability for some
    number of classes."""

    def _process_model_outputs(self, inputs, scores):
        """Processes and validates a list of model outputs.

        This is a task-dependent operation. For example, classification
        outputs need to have a softmax applied.
        """
        # Automatically cast a list or ndarray of predictions to a tensor.
        if isinstance(scores, list) or isinstance(scores, np.ndarray):
            scores = torch.tensor(scores)

        # Ensure the returned value is now a tensor.
        if not isinstance(scores, torch.Tensor):
            raise TypeError(
                "Must have list, np.ndarray, or torch.Tensor of "
                f"scores. Got type {type(scores)}"
            )

        # Validation check on model score dimensions
        # if scores.ndim == 1:
        #     # Unsqueeze prediction, if it's been squeezed by the model.
        #     if len(inputs) == 1:
        #         scores = scores.unsqueeze(dim=0)
        #     else:
        #         raise ValueError(
        #             f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
        #         )
        # elif scores.ndim != 2:
        #     # If model somehow returns too may dimensions, throw an error.
        #     raise ValueError(
        #         f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
        #     )
        # elif scores.shape[0] != len(inputs):
        #     # If model returns an incorrect number of scores, throw an error.
        #     raise ValueError(
        #         f"Model return score of shape {scores.shape} for {len(inputs)} inputs."
        #     )
        # elif not ((scores.sum(dim=1) - 1).abs() < 1e-6).all():
        #     # Values in each row should sum up to 1. The model should return a
        #     # set of numbers corresponding to probabilities, which should add
        #     # up to 1. Since they are `torch.float` values, allow a small
        #     # error in the summation.
        #     scores = torch.nn.functional.softmax(scores, dim=1)
        #     if not ((scores.sum(dim=1) - 1).abs() < 1e-6).all():
        #         raise ValueError("Model scores do not add up to 1.")
        return scores.cpu()

    def _goal_function_result_type(self):
        """Returns the class of this goal function's results."""
        return SemanticGoalFunctionResult

    def extra_repr_keys(self):
        return []

    def _get_displayed_output(self, raw_output):
        return raw_output.tolist()

    def get_results(self, attacked_text_list, check_skip=False):
        """For each attacked_text object in attacked_text_list, returns a
        result consisting of whether or not the goal has been achieved, the
        output for display purposes, and a score.

        Additionally returns whether the search is over due to the query
        budget.
        """
        results = []
        if self.query_budget < float("inf"):
            queries_left = self.query_budget - self.num_queries
            if hasattr(self, 'max_single_query'):
                max_single_query=getattr(self, 'max_single_query')
                attacked_text_list = attacked_text_list[:min(queries_left, max_single_query)]
            else:
                attacked_text_list = attacked_text_list[:queries_left]
        self.num_queries += len(attacked_text_list)
        model_outputs = self._call_model(attacked_text_list)
        for attacked_text, raw_output in zip(attacked_text_list, model_outputs):
            # displayed_output = self._get_displayed_output(raw_output)
            goal_status = self._get_goal_status(
                raw_output, attacked_text, check_skip=check_skip
            )
            goal_function_score = self._get_score(raw_output, self.initial_attacked_text)
            # if goal_function_score<-1:
            #     print()
            results.append(
                self._goal_function_result_type()(
                    attacked_text,
                    self.initial_attacked_text.words_diff_num(attacked_text),
                    # displayed_output,
                    goal_status,
                    goal_function_score,
                    self.num_queries,
                    self.ground_truth_output,
                )
            )
        return results, self.num_queries == self.query_budget
