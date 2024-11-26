
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
from textattack.utils import Logger, to_string, save_json, save_jsonl
import datetime
from llm_wm import LLM_WM
import os

def get_advtext_filename(
        wm_name='',
        attack_name='',
        victim_name='',
        llm_name='',
        num_examples=0,
        file_type='.json'
    ):

    return '_'.join([
        wm_name, attack_name, victim_name.replace('/','_'), llm_name.replace('/','_'), 
        str(num_examples), str(datetime.datetime.now().date())
    ])+file_type
    
class SemanticAttack:

    def __init__(
        self,
        target_cos=0.3,
        edit_distance=3,
        query_budget=500,
        attack_name = 'TextBuggerLi2018',
        victim_name = 'sentence-transformers/all-distilroberta-v1',#'sentence-transformers/all-mpnet-base-v2'#
        llm_name="facebook/opt-1.3b",
        wm_name='TS',
        logger=None,
        wm_detector=None,
        temperature=30,
    ):
        self.target_cos=target_cos
        self.edit_distance=edit_distance
        self.query_budget=query_budget
        self.attack_name=attack_name
        self.victim_name=victim_name
        self.llm_name=llm_name
        self.wm_name=wm_name
        self.temperature=temperature

        self.model = transformers.AutoModel.from_pretrained(self.victim_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.victim_name)
        self.model_wrapper = textattack.models.wrappers.HuggingFaceEncoderWrapper(self.model, self.tokenizer)
        self.temperature=temperature
        self.attack = getattr(textattack.attack_sems, self.attack_name).build(
            self.model_wrapper, 
            target_cos=self.target_cos, 
            edit_distance=self.edit_distance, 
            query_budget=self.query_budget,
            temperature=self.temperature
        )
        
        if logger is None:
            self.log=Logger(
                'attack_log/SemanticAttack'+'-'.join([
                    self.wm_name, self.attack_name, 
                    # self.victim_name.replace('/','_'), self.llm_name.replace('/','_'),
                    # str(self.temperature)
                ])+'-'+str(datetime.datetime.now())[0:-10]+'.log',
                level='debug', 
                screen=False
            )
        else:
            self.log=logger
        self.log_info('\n')

        self.result_list=[]
        self.wm_detector=wm_detector

        self.log_info(['target_cos', self.target_cos])
        self.log_info(['edit_distance', self.edit_distance])
        self.log_info(['query_budget', self.query_budget])
        self.log_info(['attack_name', self.attack_name])
        self.log_info(['victim_name', self.victim_name])
        self.log_info(['llm_name', self.llm_name])
        self.log_info(['wm_name', self.wm_name])
        self.log_info(['temperature', self.temperature])

    
    def save(self):
        filename=get_advtext_filename(
            wm_name=self.wm_name,
            llm_name=self.llm_name,
            attack_name=self.attack_name,
            victim_name=self.victim_name,
            num_examples=len(self.result_list)
        )
        file_path=os.path.join('saved_data', filename)
        save_json(data=self.result_list, file_path=file_path)

    def log_info(self, info=''):
        if not isinstance(info, str):
            info=to_string(info)
        self.log.logger.info(info)
        # print(info)

    def iter_get_adv(self, sentence, sen_detect={}, ground_truth_output=0, window_size=10, step_ize=10, attack_times=3, step_target=0.7):
        return

    def get_adv(self, sentence, sen_detect={}, ground_truth_output=0, window_size=10, step_ize=10, attack_times=3):
        for idx in range(attack_times):
            attacked=self.attack.step_attack(sentence, ground_truth_output, window_size=window_size, step_ize=step_ize)
            overall_score=round(np.mean(attacked['overall_score']),4)
            simi_score=round(np.mean(attacked['score']),4)
            num_queries=round(np.mean(attacked['num_queries']),3)
            budget=np.sum(attacked['budget'])
            attacked_rlt=self.wm_detector(attacked['text'])
            if attacked_rlt['is_watermarked']==False:
                break
            else:
                self.log_info(['Failed', idx, 'simi_score', attacked['score']])
                self.log_info(['Failed', idx, 'adv_detect',attacked_rlt])


        self.log_info(['adv_text:', attacked['text'].replace('\n',' ')])
        self.log_info(['overall_score', overall_score])
        self.log_info(['simi_score', attacked['score']])
        self.log_info(['num_queries', num_queries])
        self.log_info(['budget', budget])
        self.log_info(['adv_detect',attacked_rlt])
        self.log_info()

        self.result_list.append({
            'raw_text': sentence,
            'raw_detect': (str(sen_detect['is_watermarked']), round(sen_detect['score'],4)),
            'adv_text': attacked['text'],
            'overall_score': attacked['overall_score'],
            'simi_score': attacked['score'],
            'num_queries': attacked['num_queries'],
            'budget': attacked['budget'],
            'adv_detect': (str(attacked_rlt['is_watermarked']), round(attacked_rlt['score'],4))
        })
        return attacked_rlt['is_watermarked'], attacked['overall_score'], num_queries, budget