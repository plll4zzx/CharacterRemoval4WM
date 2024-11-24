
import torch
import json
from MarkLLM.watermark.auto_watermark import AutoWatermark
from MarkLLM.utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

import textattack
from read_data import c4
import textattack.attack_sems
import numpy as np
from textattack.utils import Logger, to_string
import datetime

class LLM_WM:

    def __init__(self, model_name = "facebook/opt-1.3b", device = "cuda", wm_name='KGW'):
        self.model_name = model_name
        self.wm_name=wm_name
        self.device = device if torch.cuda.is_available() else "cpu"

        # Transformers config
        self.transformers_config = TransformersConfig(
            model=AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device),
            tokenizer=AutoTokenizer.from_pretrained(self.model_name),
            vocab_size=50272,
            device=self.device,
            max_new_tokens=200,
            min_length=230,
            do_sample=True,
            no_repeat_ngram_size=4
        )

        # Load watermark algorithm
        self.wm_model = AutoWatermark.load(f'{self.wm_name}', algorithm_config=f'MarkLLM/config/{self.wm_name}.json', transformers_config=self.transformers_config)
    
    def generate(self, prompt, un_wm_flag=False, wm_seed=None, un_wm_seed=None):
        # Generate text
        if wm_seed is not None:
            torch.manual_seed(wm_seed)
        wm_text = self.wm_model.generate_watermarked_text(prompt)
        if un_wm_seed is not None:
            torch.manual_seed(un_wm_seed)
        if un_wm_flag:
            un_wm_text = self.wm_model.generate_unwatermarked_text(prompt)
        else:
            un_wm_text=''
        return wm_text, un_wm_text

    def detect_wm(self, text):
        # Detect
        result = self.wm_model.detect_watermark(text)
        return result