
import numpy as np
import torch
from torch.nn.functional import softmax

from textattack.goal_function_results import GoalFunctionResultStatus, SemanticGoalFunctionResult
from textattack.search_methods import SearchMethod
from textattack.shared.validators import (
    transformation_consists_of_word_swaps_and_deletions,
)
from .greedy_word_swap_wir import GreedyWordSwapWIR
from copy import deepcopy

class SlidingWindowWSample(GreedyWordSwapWIR):
    def __init__(self, wir_method="unk", unk_token="[UNK]", temperature=30, window_size=10, step_size=1):
        self.wir_method = wir_method
        self.unk_token = unk_token
        self.temperature=temperature
        self.window_size = window_size 
        self.step_size = step_size
    
    def get_sliding_windows(self, initial_text, indices_to_order):
        len_words=len(initial_text.words)
        leave_one_texts={}
        for idx in indices_to_order:
            if idx>=self.window_size and idx<=(len_words-self.window_size):
                leave_one_texts[idx]=[]
                for idy in range(self.window_size):
                    tmp_text=initial_text.delete_word_at_index(idx)
                    tmp_words=deepcopy(tmp_text.words)
                    for idz in range(len(tmp_text.words)):
                        if idz not in range(idx-self.window_size+idy-1,idx+idy):
                            tmp_words[idz]=''
                    tmp_text=tmp_text.generate_new_attacked_text(tmp_words)
                    leave_one_texts[idx].append(tmp_text)
            elif idx<self.window_size:
                leave_one_texts[idx]=[]
                for idy in range(idx+1):
                    tmp_text=initial_text.delete_word_at_index(idx)
                    tmp_words=deepcopy(tmp_text.words)
                    for idz in range(len(tmp_text.words)):
                        if idz not in range(idx-idy, idx+self.window_size-idy):
                            tmp_words[idz]=''
                    tmp_text=tmp_text.generate_new_attacked_text(tmp_words)
                    leave_one_texts[idx].append(tmp_text)
                # tmp_words=deepcopy(initial_text.words)
                # for idz in range(len(tmp_words)):
                #     if idz not in range(0,self.window_size):
                #         tmp_words[idz]=''
                # tmp_text=initial_text.generate_new_attacked_text(tmp_words)
                # tmp_text=tmp_text.delete_word_at_index(idx)
                # leave_one_texts[idx]=[tmp_text]
            elif idx>(len_words-self.window_size):
                leave_one_texts[idx]=[]
                for idy in range(len_words-idx):
                    tmp_text=initial_text.delete_word_at_index(idx)
                    tmp_words=deepcopy(tmp_text.words)
                    for idz in range(len(tmp_text.words)):
                        if idz not in range(idx-self.window_size+idy-1, len_words):
                            tmp_words[idz]=''
                    tmp_text=tmp_text.generate_new_attacked_text(tmp_words)
                    leave_one_texts[idx].append(tmp_text)
            #     tmp_text=initial_text.delete_word_at_index(idx)
            #     tmp_words=deepcopy(tmp_text.words)
            #     for idz in range(len(tmp_text.words)):
            #         if idz not in range(-(self.window_size-1), len(tmp_words)):
            #             tmp_words[idz]=''
            #     tmp_text=tmp_text.generate_new_attacked_text(tmp_words)
            #     leave_one_texts[idx]=[tmp_text]
        return leave_one_texts

    def _get_index_order(self, initial_text, max_len=-1):
        """Returns word indices of ``initial_text`` in descending order of
        importance."""

        len_text, indices_to_order = self.get_indices_to_order(initial_text)

        if "unk" in self.wir_method:
            
            leave_one_texts = [
                initial_text.replace_word_at_index(i, self.unk_token)
                for i in indices_to_order
            ]
            leave_one_results, search_over = self.get_goal_results(leave_one_texts)
            index_scores = np.array([result.score for result in leave_one_results])

        elif "weighted-saliency" in self.wir_method:
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

        elif "delete" in self.wir_method:
            leave_one_texts = self.get_sliding_windows(initial_text, indices_to_order)
            search_overs=np.zeros(max(indices_to_order)+1)
            index_scores=np.zeros(max(indices_to_order)+1)
            for idx in leave_one_texts:
                tmp_results, tmp_search_over=self.get_goal_results(leave_one_texts[idx])
                index_scores[idx]=np.mean([result.score for result in tmp_results])
                search_overs[idx]=int(tmp_search_over)
            search_over=np.mean(search_overs)>0.5
                
        elif "gradient" in self.wir_method:
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

        elif "random" in self.wir_method:
            index_order = indices_to_order
            np.random.shuffle(index_order)
            search_over = False
        else:
            raise ValueError(f"Unsupported WIR method {self.wir_method}")


        if "random" not in self.wir_method:
        #     index_order = np.array(indices_to_order)[(-index_scores).argsort()]
        # else:
            distri = softmax(
                torch.Tensor(index_scores*self.temperature), dim=0
            ).numpy()
            index_order=np.random.choice(len(distri), size=min(10, int(len(distri)/2)), p=distri, replace=False)

        return index_order, search_over

    def perform_search(self, initial_result):
        attacked_text = initial_result.attacked_text

        # Sort words by order of importance
        index_order, search_over = self._get_index_order(attacked_text)
        i = 0
        cur_result = initial_result
        results = None
        while i < len(index_order) and not search_over:
            transformed_text_candidates = self.get_transformations(
                cur_result.attacked_text,
                original_text=initial_result.attacked_text,
                indices_to_modify=[index_order[i]],
            )
            i += 1
            if len(transformed_text_candidates) == 0:
                continue
            results, search_over = self.get_goal_results(transformed_text_candidates)
            results = sorted(results, key=lambda x: -x.score)
            # Skip swaps which don't improve the score
            if results[0].score > cur_result.score:
                cur_result = results[0]
            else:
                continue
            # If we succeeded, return the index with best similarity.
            if cur_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                best_result = cur_result
                # @TODO: Use vectorwise operations
                max_similarity = -float("inf")
                for result in results:
                    if result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        break
                    candidate = result.attacked_text
                    try:
                        similarity_score = candidate.attack_attrs["similarity_score"]
                    except KeyError:
                        # If the attack was run without any similarity metrics,
                        # candidates won't have a similarity score. In this
                        # case, break and return the candidate that changed
                        # the original score the most.
                        break
                    if similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_result = result
                return best_result

        return cur_result