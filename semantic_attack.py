
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
from textattack.utils import Logger, to_string, save_json
import datetime
from llm_wm import LLM_WM

def get_advtext_filename(
        attack_name='',
        dataset_name='',
        victim_name='',
        num_examples=0,
        file_type='.json'
    ):

    return '_'.join([
        attack_name, dataset_name.replace('/','_'), victim_name.replace('/','_'), str(num_examples)
    ])+file_type
    
class SemanticAttack:

    def __init__(
        self,
        target_cos=0.3,
        edit_distance=3,
        query_budget=500,
        attack_name = 'TextBuggerLi2018',
        victim_name = 'sentence-transformers/all-distilroberta-v1',#'sentence-transformers/all-mpnet-base-v2'#
        logger=None,
        wm_detector=None,
    ):
        self.target_cos=target_cos
        self.edit_distance=edit_distance
        self.query_budget=query_budget
        self.attack_name=attack_name
        self.victim_name=victim_name

        self.model = transformers.AutoModel.from_pretrained(self.victim_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.victim_name)
        self.model_wrapper = textattack.models.wrappers.HuggingFaceEncoderWrapper(self.model, self.tokenizer)
        self.attack = getattr(textattack.attack_sems, self.attack_name).build(
            self.model_wrapper, 
            target_cos=self.target_cos, 
            edit_distance=self.edit_distance, 
            query_budget=self.query_budget
        )
        
        if logger is None:
            self.log=Logger(
                'attack_log/SemanticAttack'+'-'+str(datetime.datetime.now().date())+'.log',
                level='debug', 
                screen=False
            )
        else:
            self.log=logger
        self.log_info('\n')

        self.result_list=[]
        self.wm_detector=wm_detector

    
    def __del__(self):
        import os
        filename=get_advtext_filename(
            attack_name=self.attack_name,
            victim_name=self.victim_name,
        )
        file_path=os.path.join('saved_data', filename)
        save_json(data=self.result_list, file_path=file_path)

    def log_info(self, info):
        if not isinstance(info, str):
            info=to_string(info)
        self.log.logger.info(info)
        print(info)

    def get_adv(self, sentence, ground_truth_output=0, window_size=10, step_ize=10):
        attacked=self.attack.step_attack(sentence, ground_truth_output, window_size=window_size, step_ize=step_ize)
        simi_score=round(np.mean(attacked['score']),4)
        num_queries=round(np.mean(attacked['num_queries']),3)
        budget=np.sum(attacked['budget'])
        self.log_info(['ATTACKED:', attacked['text'].replace('\n',' ')])
        self.log_info(['simi_score', simi_score, '; num_queries', num_queries, '; budget', budget])
        attacked_rlt=self.wm_detector.detect_wm(attacked['text'])
        self.log_info(['attacked',attacked_rlt])
        self.log_info()

        self.result_list.append({
            'raw_text': sentence,
            'adv_text': attacked['text'],
            'simi_score': attacked['score'],
            'num_queries': attacked['num_queries'],
            'budget': attacked['budget'],
            'adv_detect': attacked_rlt
        })