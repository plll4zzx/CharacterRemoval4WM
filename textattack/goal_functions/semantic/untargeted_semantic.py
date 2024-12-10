"""
Goal Function for Attempts to minimize the BLEU score
-------------------------------------------------------


"""

import functools

import nltk

import textattack

from .semantic_goal_function import SemanticGoalFunctionResult,SemanticGoalFunction
# from ...constraints.semantics.sentence_encoders.sentence_bert.sbert import SBERT
# from ...constraints.semantics.sentence_encoders.universal_sentence_encoder.universal_sentence_encoder import UniversalSentenceEncoder

import torch
from torch import Tensor, device

class UntargetedSemantic(SemanticGoalFunction):
    """
    """

    EPS = 1e-10

    def __init__(self, *args, target_cos=0.0, max_single_query=100,**kwargs):
        self.target_cos = target_cos
        super().__init__(*args, **kwargs)
        self.gt_embedding=None
        self.gt_tokenizer_input=''
        self.max_single_query=max_single_query

    def clear_cache(self):
        if self.use_cache:
            self._call_model_cache.clear()
        # get_bleu.cache_clear()

    def _is_goal_complete(self, model_output, _):
        neg_cos_score = self._get_score(model_output, self.initial_attacked_text)
        return neg_cos_score >= (1-self.target_cos) - UntargetedSemantic.EPS

    def _get_score(self, model_output, initial_attacked_text):
        if isinstance(model_output,str) and isinstance(initial_attacked_text,str):
            cos_score = cos_sim(
                self.model(model_output), 
                self.model(initial_attacked_text)
            )
        else:
            if initial_attacked_text.tokenizer_input!=self.gt_tokenizer_input:
                gt_embedding = self.model(initial_attacked_text.tokenizer_input)
                self.gt_tokenizer_input=initial_attacked_text.tokenizer_input
                self.gt_embedding = gt_embedding.cpu()
            # gt_embedding = gt_embedding.cpu()+torch.normal(mean=0, std=0.1, size=gt_embedding.shape)
            cos_score = cos_sim(model_output, self.gt_embedding)
        return 1-cos_score


# @functools.lru_cache(maxsize=2**12)
def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))
