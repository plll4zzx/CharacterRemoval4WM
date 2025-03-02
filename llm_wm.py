
import torch
from MarkLLM.watermark.auto_watermark import AutoWatermark
from MarkLLM.utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from textattack.utils import Logger, to_string
# import json
# import transformers
# import textattack
# from read_data import c4
# import textattack.attack_sems
# import numpy as np
# import datetime
# from optimum.bettertransformer import BetterTransformer

class LLM_WM:

    def __init__(self, model_name = "facebook/opt-1.3b", device = "cuda", wm_name='KGW'):
        self.model_name = model_name
        self.wm_name=wm_name
        self.device = device if torch.cuda.is_available() else "cpu"
        tokenizer=AutoTokenizer.from_pretrained(self.model_name)
        if "llama" in model_name.lower():
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype="auto",
                quantization_config={
                    "load_in_4bit":True, 
                    # "load_in_8bit": True,  
                    "bnb_4bit_compute_dtype": "float16",  # 指定计算精度
                    "bnb_4bit_use_double_quant": True,   # 是否使用双量化
                    "bnb_4bit_quant_type": "nf4"         # 量化类型（nf4 通常表现更好）
                }
            )
            tokenizer.pad_token = "[PAD]"
            model.config.pad_token_id = tokenizer.eos_token_id
            # model = BetterTransformer.transform(model)#.to(self.device)
        else:
            model=AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        if "opt" in model_name.lower():
            vocab_size=50272
        else:
            vocab_size=len(tokenizer)
        model.config.pad_token_id = tokenizer.eos_token_id  
        model.config.eos_token_id = tokenizer.eos_token_id  
        # Transformers config
        self.transformers_config = TransformersConfig(
            model=model,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            device=self.device,
            max_new_tokens=350,
            min_length=300,
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
        if not isinstance(result['is_watermarked'], bool):
            result['is_watermarked']=bool(result['is_watermarked'])
        if result['score'] is None:
            if result['is_watermarked']:
                result['score']=1
            else:
                result['score']=0
        return result
    
    def get_perplexity(self, sentence):
        encodings = self.transformers_config.tokenizer(sentence, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.transformers_config.model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss  # 交叉熵损失

        perplexity = torch.exp(loss)  # PPL = e^loss
        return perplexity.item()