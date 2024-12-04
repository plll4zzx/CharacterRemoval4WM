# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================
# SemStamp.py
# Description: Implementation of SemStamp algorithm
# ============================================

import torch
from math import sqrt
from functools import partial
from ..base import BaseWatermark
from MarkLLM.utils.utils import load_config_file
from MarkLLM.utils.transformers_config import TransformersConfig
from MarkLLM.exceptions.exceptions import AlgorithmNameMismatchError
from transformers import LogitsProcessor, LogitsProcessorList
from MarkLLM.visualize.data_for_visualization import DataForVisualization
import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
import os
# from sentence_transformers import SentenceTransformer
from copy import deepcopy
from scipy.spatial.distance import hamming, cosine
from nearpy.hashes import RandomBinaryProjections
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import numpy as np
from typing import List, Tuple, Callable, Optional, Iterator
global Device
import transformers
import textattack

def batched(iterable, n, total=None):
    l = len(iterable)
    if total is not None:
        assert l == total
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class LSHModel:
    def __init__(self, device, batch_size, lsh_dim):
        self.comparator: Callable[[np.ndarray, np.ndarray], float]
        self.hasher = None
        self.do_lsh: bool = False
        self.dimension: int = -1
        self.device = device
        self.batch_size: int = batch_size
        self.lsh_dim: int = lsh_dim
        print("initializing random projection LSH model")
        self.hasher = RandomBinaryProjections(
            'rbp_perm', projection_count=self.lsh_dim, rand_seed=1234)
        self.do_lsh = True
        self.comparator = lambda x, y: hamming(*[
            np.fromstring(self.hasher.hash_vector(i)[0], 'u1') - ord('0')
            for i in [x, y]])
        self.comparator = lambda x, y: cosine(x, y)

    def compute_distances(self, refs: List[str], cands: List[str]) -> np.ndarray:
        '''
        :param refs: list of reference sentences
        :param cands: list of candidate sentences to compute similarity distances from references
        :return:
        '''
        assert len(refs) == len(cands)
        results = np.zeros(len(refs))
        i = 0
        for batch in batched(zip(refs, cands), self.batch_size, total=len(refs)):
            (ref_b, cands_b) = list(zip(*batch))
            assert len(ref_b) <= self.batch_size
            [ref_features, cand_features] = [
                self.get_embeddings(x) for x in [ref_b, cands_b]]

            if i == 0:
                print(
                    f"comparing vectors of dimension {ref_features.shape[-1]}")
            results[i:i + len(ref_b)] = np.fromiter(
                map(lambda args: self.comparator(*args), zip(ref_features, cand_features)), dtype=float)
            i += len(ref_b)

        return results

    def get_embeddings(self, sents: Iterator[str]) -> np.ndarray:
        '''
        retrieve np array of sentence embeddings from sentence iterator
        :param sents: set of sentence strings
        :return: extracted embeddings
        '''
        raise NotImplementedError()

    def get_hash(self, sents: Iterator[str]) -> Iterator[str]:
        embd = self.get_embeddings(sents)
        # print(f"embedding: {embd}")
        hash_strs = [self.hasher.hash_vector(e)[0] for e in embd]
        hash_ints = [int(s, 2) for s in hash_strs]
        return hash_ints


class SBERTLSHModel(LSHModel):
    def __init__(self, device, batch_size, lsh_dim, sbert_type='roberta', lsh_model_path=None, **kwargs):
        super(SBERTLSHModel, self).__init__(device, batch_size, lsh_dim)
        self.sbert_type = sbert_type
        self.dimension = 1024 if 'large' in self.sbert_type else 768

        print(f'loading SBERT {self.sbert_type} model...')
        # self.embedder = SentenceTransformer(f"{OPTS.sbert_type}-nli-mean-tokens")
        # try:
        # if lsh_model_path is not None:
        #     self.embedder = SentenceTransformer(lsh_model_path)
        #     self.dimension = self.embedder.get_sentence_embedding_dimension()
        # else:
        #     self.embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v1")
        # # except:
        # #     self.embedder = SentenceTransformer(f"{os.getenv('HOME')}/.cache/torch/sentence_transformers/sentence-transformers_{self.sbert_type}-nli-stsb-mean-tokens")
        # # self.embedder.eval()
        # # self.device.move(self.embedder)
        # self.embedder = self.embedder.to(self.device)

        model = transformers.AutoModel.from_pretrained(
            lsh_model_path, 
            # torch_dtype=torch.float16
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(lsh_model_path)
        self.embedder = textattack.models.wrappers.HuggingFaceEncoderWrapper(model, tokenizer)
        self.embedder.model.eval()

        self.hasher.reset(dim=self.dimension)

    def get_embeddings(self, sents: Iterator[str]) -> np.ndarray:
        # all_embeddings = self.embedder.encode(sents, batch_size=self.batch_size)
        all_embeddings = self.embedder(sents).cpu()
        return np.stack(all_embeddings)

class SemStampConfig:
    """Config class for SemStamp algorithm, load config file and initialize parameters."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the SemStamp configuration.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        if algorithm_config is None:
            config_dict = load_config_file('config/SemStamp.json')
        else:
            config_dict = load_config_file(algorithm_config)
        if config_dict['algorithm_name'] != 'SemStamp':
            raise AlgorithmNameMismatchError('SemStamp', config_dict['algorithm_name'])

        self.gamma = config_dict['gamma']
        self.delta = config_dict['delta']
        self.hash_key = config_dict['hash_key']
        self.z_threshold = config_dict['z_threshold']
        self.prefix_length = config_dict['prefix_length']
        self.f_scheme = config_dict['f_scheme']
        self.window_scheme = config_dict['window_scheme']
        self.lsh_model = SBERTLSHModel(lsh_model_path=config_dict['embedder'], device=config_dict['device'], batch_size=1, lsh_dim=config_dict['lsh_dim'], sbert_type=config_dict['sbert_type'])

        self.generation_model = transformers_config.model
        self.generation_tokenizer = transformers_config.tokenizer
        self.vocab_size = transformers_config.vocab_size
        self.device = transformers_config.device
        self.gen_kwargs = transformers_config.gen_kwargs

class SemStampUtils:
    """Utility class for SemStamp algorithm, contains helper functions."""

    def __init__(self, config: SemStampConfig, *args, **kwargs) -> None:
        """
            Initialize the SemStamp utility class.

            Parameters:
                config (SemStampConfig): Configuration for the SemStamp algorithm.
        """
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(self.config.hash_key)
        self.prf = torch.randperm(self.config.vocab_size, device=self.config.device, generator=self.rng)
        self.f_scheme_map = {"time": self._f_time, "additive": self._f_additive, "skip": self._f_skip, "min": self._f_min}
        self.window_scheme_map = self._get_greenlist_ids

    def _f(self, input_ids: torch.LongTensor) -> int:
        """Get the previous token."""
        return int(self.f_scheme_map[self.config.f_scheme](input_ids))
    
    def _f_time(self, input_ids: torch.LongTensor):
        """Get the previous token time."""
        time_result = 1
        for i in range(0, self.config.prefix_length):
            time_result *= input_ids[-1 - i].item()
        return self.prf[time_result % self.config.vocab_size]
    
    def _f_additive(self, input_ids: torch.LongTensor):
        """Get the previous token additive."""
        additive_result = 0
        for i in range(0, self.config.prefix_length):
            additive_result += input_ids[-1 - i].item()
        return self.prf[additive_result % self.config.vocab_size]
    
    def _f_skip(self, input_ids: torch.LongTensor):
        """Get the previous token skip."""
        return self.prf[input_ids[- self.config.prefix_length].item()]

    def _f_min(self, input_ids: torch.LongTensor):
        """Get the previous token min."""
        return min(self.prf[input_ids[-1 - i].item()] for i in range(0, self.config.prefix_length))
    
    def get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids."""
        return self.window_scheme_map(input_ids)
    
    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        """Get greenlist ids for the input_ids via leftHash scheme."""
        lsh_seed = self.config.lsh_model.get_hash([self.config.generation_tokenizer.decode(input_ids)])[0]
        # n_bins = 2**self.config.lsh_model.lsh_dim
        # n_accept = int(n_bins * self.config.gamma)
        # self.rng.manual_seed(self.config.hash_key * lsh_seed)
        # vocab_permutation = torch.randperm(n_bins, device=self.config.device, generator=self.rng)
        # greenlist_ids = vocab_permutation[:n_accept]


        self.rng.manual_seed((self.config.hash_key * lsh_seed) % self.config.vocab_size)
        greenlist_size = int(self.config.vocab_size * self.config.gamma)
        vocab_permutation = torch.randperm(self.config.vocab_size, device=input_ids.device, generator=self.rng)
        greenlist_ids = vocab_permutation[:greenlist_size]
        return greenlist_ids
    
    def _compute_z_score(self, observed_count: int , T: int) -> float: 
        """Compute z-score for the given observed count and total tokens."""
        expected_count = self.config.gamma
        numer = observed_count - expected_count * T 
        denom = sqrt(T * expected_count * (1 - expected_count))  
        z = numer / denom
        return z
    
    def score_sequence(self, input_ids: torch.Tensor) -> tuple[float, list[int]]:
        """Score the input_ids and return z_score and green_token_flags."""
        num_tokens_scored = len(input_ids) - self.config.prefix_length
        if num_tokens_scored < 1:
            raise ValueError(
                (
                    f"Must have at least {1} token to score after "
                    f"the first min_prefix_len={self.config.prefix_length} tokens required by the seeding scheme."
                )
            )

        green_token_count = 0
        green_token_flags = [-1 for _ in range(self.config.prefix_length)]

        for idx in range(self.config.prefix_length, len(input_ids)):
            curr_token = input_ids[idx]
            greenlist_ids = self.get_greenlist_ids(input_ids[:idx])
            if curr_token in greenlist_ids:
                green_token_count += 1
                green_token_flags.append(1)
            else:
                green_token_flags.append(0)
        
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        return z_score, green_token_flags


class SemStampLogitsProcessor(LogitsProcessor):
    """LogitsProcessor for SemStamp algorithm, process logits to add watermark."""

    def __init__(self, config: SemStampConfig, utils: SemStampUtils, *args, **kwargs) -> None:
        """
            Initialize the SemStamp logits processor.

            Parameters:
                config (SemStampConfig): Configuration for the SemStamp algorithm.
                utils (SemStampUtils): Utility class for the SemStamp algorithm.
        """
        self.config = config
        self.utils = utils

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids: torch.LongTensor) -> torch.BoolTensor:
        """Calculate greenlist mask for the given scores and greenlist token ids."""
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Bias the scores for the greenlist tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits to add watermark."""
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self.utils.get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)

        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.config.delta)
        return scores
    

class SemStamp(BaseWatermark):
    """Top-level class for SemStamp algorithm."""

    def __init__(self, algorithm_config: str, transformers_config: TransformersConfig, *args, **kwargs) -> None:
        """
            Initialize the SemStamp algorithm.

            Parameters:
                algorithm_config (str): Path to the algorithm configuration file.
                transformers_config (TransformersConfig): Configuration for the transformers model.
        """
        self.config = SemStampConfig(algorithm_config, transformers_config)
        self.utils = SemStampUtils(self.config)
        self.logits_processor = SemStampLogitsProcessor(self.config, self.utils)
    
    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        """Generate watermarked text."""

        # Configure generate_with_watermark
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]), 
            **self.config.gen_kwargs
        )
        
        # Encode prompt
        encoded_prompt = self.config.generation_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True, 
            # padding=True, truncation=True
        ).to(self.config.device)
        # Generate watermarked text
        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        # Decode
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]
        return watermarked_text
    
    def detect_watermark(self, text: str, return_dict: bool = True, *args, **kwargs):
        """Detect watermark in the text."""

        # Encode the text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)

        # Compute z_score using a utility method
        z_score, _ = self.utils.score_sequence(encoded_text)

        # Determine if the z_score indicates a watermark
        is_watermarked = z_score > self.config.z_threshold

        # Return results based on the return_dict flag
        if return_dict:
            return {"is_watermarked": is_watermarked, "score": z_score}
        else:
            return (is_watermarked, z_score)
        
    def get_data_for_visualization(self, text: str, *args, **kwargs) -> tuple[list[str], list[int]]:
        """Get data for visualization."""
        
        # Encode text
        encoded_text = self.config.generation_tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.config.device)
        
        # Compute z-score and highlight values
        z_score, highlight_values = self.utils.score_sequence(encoded_text)
        
        # decode single tokens
        decoded_tokens = []
        for token_id in encoded_text:
            token = self.config.generation_tokenizer.decode(token_id.item())
            decoded_tokens.append(token)
        
        return DataForVisualization(decoded_tokens, highlight_values)