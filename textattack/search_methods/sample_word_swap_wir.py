
import numpy as np
import torch
from torch.nn.functional import softmax

from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)
from .greedy_word_swap_wir import GreedyWordSwapWIR



class SampleWordSwapWIR(GreedyWordSwapWIR):
    def __init__(self, wir_method="unk", unk_token="[UNK]", temperature=30):
        self.wir_method = wir_method
        self.unk_token = unk_token
        self.temperature=temperature

    def _get_index_order(self, initial_text, max_len=-1):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""

        len_text, indices_to_order = self.get_indices_to_order(initial_text)

        if self.wir_method == "unk":
            leave_one_texts = [
                initial_text.replace_word_at_index(i, self.unk_token)
                for i in indices_to_order
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "weighted-saliency":
            # first, compute word saliency
            leave_one_texts = [
                initial_text.replace_word_at_index(i, self.unk_token)
                for i in indices_to_order
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            saliency_scores = np.array([result.score for result in leave_one_results])

            softmax_saliency_scores = softmax(
                torch.Tensor(saliency_scores), dim=0
            ).numpy()

            # compute the largest change in score we can find by swapping each word
            delta_ps = []
            for idx in indices_to_order:
                # Exit Loop when search_over is True - but we need to make sure delta_ps
                # is the same size as softmax_saliency_scores
                if search_over:
                    delta_ps = delta_ps + [0.0] * (
                        len(softmax_saliency_scores) - len(delta_ps)
                    )
                    break

                transformed_text_candidates = self.get_transformations(
                    initial_text,
                    original_text=initial_text,
                    indices_to_modify=[idx],
                )
                if not transformed_text_candidates:
                    # no valid synonym substitutions for this word
                    delta_ps.append(0.0)
                    continue
                swap_results, search_over = self.get_goal_results(
                    transformed_text_candidates
                )
                score_change = [result.score for result in swap_results]
                if not score_change:
                    delta_ps.append(0.0)
                    continue
                max_score_change = np.max(score_change)
                delta_ps.append(max_score_change)

            index_scores = softmax_saliency_scores * np.array(delta_ps)

        elif self.wir_method == "delete":
            leave_one_texts = [
                initial_text.delete_word_at_index(i) for i in indices_to_order
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif self.wir_method == "gradient":
            victim_model = self.get_victim_model()
            index_scores = np.zeros(len_text)
            grad_output = victim_model.get_grad(initial_text.tokenizer_input)
            gradient = grad_output["gradient"]
            word2token_mapping = initial_text.align_with_model_tokens(victim_model)
            for i, index in enumerate(indices_to_order):
                matched_tokens = word2token_mapping[index]
                if not matched_tokens:
                    index_scores[i] = 0.0
                else:
                    agg_grad = np.mean(gradient[matched_tokens], axis=0)
                    index_scores[i] = np.linalg.norm(agg_grad, ord=1)

            search_over = False

        elif self.wir_method == "random":
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")


        if self.wir_method != "random":
            # index_order = np.array(indices_to_order)[(-index_scores).argsort()]
            distri = softmax(
                    torch.Tensor(index_scores*self.temperature), dim=0
                ).numpy()
            tmp_order=np.random.choice(len(distri), size=10, p=distri, replace=False)
            index_order=np.zeros_like(tmp_order)
            for idx in range(len(tmp_order)):
                index_order[idx]=indices_to_order[tmp_order[idx]]

        return index_order, search_over