
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
from textattack.utils import Logger, to_string, save_json, save_jsonl, truncation
import datetime
from llm_wm import LLM_WM
import os
from tqdm import tqdm

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
        sep_size=None,
        query_budget=500,
        random_num=5, random_one=True,
        attack_name = 'TextBuggerLi2018',
        victim_name = 'sentence-transformers/all-distilroberta-v1',#'sentence-transformers/all-mpnet-base-v2'#
        llm_name="facebook/opt-1.3b",
        wm_name='TS',
        logger=None,
        wm_detector=None,
        temperature=30,
        max_single_query=20,
        slide_flag=True
    ):
        self.target_cos=target_cos
        self.edit_distance=edit_distance
        self.sep_size=sep_size
        self.query_budget=query_budget
        self.attack_name=attack_name
        self.victim_name=victim_name
        self.llm_name=llm_name
        self.wm_name=wm_name
        self.temperature=temperature
        self.random_num=random_num 
        self.random_one=random_one
        self.max_single_query=max_single_query
        self.slide_flag=slide_flag

        if not self.slide_flag:
            self.max_single_query=self.query_budget

        self.model = transformers.AutoModel.from_pretrained(
            self.victim_name, 
            # torch_dtype=torch.float16
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.victim_name)
        self.model_wrapper = textattack.models.wrappers.HuggingFaceEncoderWrapper(self.model, self.tokenizer)
        self.temperature=temperature
        self.attack = getattr(textattack.attack_sems, self.attack_name).build(
            self.model_wrapper, 
            target_cos=self.target_cos, 
            edit_distance=self.edit_distance, 
            query_budget=self.query_budget,
            temperature=self.temperature,
            max_single_query=self.max_single_query,
            random_num=self.random_num, 
            random_one=self.random_one,
            slide_flag=slide_flag
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
        self.log_info(['sep_size', self.sep_size])
        self.log_info(['query_budget', self.query_budget])
        self.log_info(['attack_name', self.attack_name])
        self.log_info(['victim_name', self.victim_name])
        self.log_info(['llm_name', self.llm_name])
        self.log_info(['wm_name', self.wm_name])
        self.log_info(['temperature', self.temperature])
        self.log_info(['max_single_query', self.max_single_query])
        self.log_info(['slide_flag', self.slide_flag])

    def truncation(self, text, max_token_num=100):
        new_text, token_num=self.attack.truncation(text, max_token_num)
        return new_text, token_num
    
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

    def iter_get_adv(
        self, sentence, sen_detect={}, ground_truth_output=0, 
        window_size=10, step_ize=10, attack_times=3, step_target=0.8
    ):
        sentence_ids=self.attack.goal_function.model.tokenizer.encode(sentence)[1:-1]
        sub_sentence_list=[
            self.attack.goal_function.model.tokenizer.decode(sentence_ids[idx:idx+window_size])
            for idx in range(0, len(sentence_ids),step_ize)
        ]
        sub_result_dict={}
        for idx in tqdm(range(len(sub_sentence_list)), leave=True, ncols=100):
            sub_result_dict[idx]=[]
            sub_sentence=sub_sentence_list[idx]
            atk_count=0
            # cos_score=1
            # ul_sub_rlt=None
            # for idy in tqdm(range(attack_times), leave=True, ncols=100):
            while atk_count<attack_times:# and cos_score>step_target:
                sub_result=self.attack.attack(sub_sentence, ground_truth_output)
                # tmp_cos_score=round(1-sub_result.perturbed_result.score,4)
                # if tmp_cos_score<=cos_score:
                #     ul_sub_rlt=sub_result
                #     cos_score=tmp_cos_score
                # self.log_info([idx, atk_count, tmp_cos_score])
                atk_count+=1
                sub_result_dict[idx].append(sub_result)
        
        rlt_text=''
        for tmp in sub_result_dict:
            tmp_text=tmp.perturbed_result.attacked_text.text
            if tmp_text[0:2]=='##':
                rlt_text=rlt_text+tmp_text[2:]
            elif tmp_text[0]==' ' or rlt_text=='' or rlt_text[-1]==' ':
                rlt_text=rlt_text+tmp_text
            else:
                rlt_text=rlt_text+' '+tmp_text
        
        attacked={
            'text': rlt_text,
            'score':[
                round(1-tmp.perturbed_result.score,4)
                for tmp in sub_result_dict
            ],
            'overall_score':round(1-self.attack.goal_function._get_score(rlt_text, sentence).item(),4),
            'num_queries':[
                tmp.perturbed_result.num_queries
                for tmp in sub_result_dict
            ],
            'budget':[
                tmp.original_result.attacked_text.words_diff_num(tmp.perturbed_result.attacked_text)
                for tmp in sub_result_dict
            ],
        }
        attacked_rlt=self.wm_detector(attacked['text'])
        num_queries=round(np.mean(attacked['num_queries']),3)
        budget=np.sum(attacked['budget'])

        self.log_info(['adv_text:', attacked['text'].replace('\n',' ')])
        self.log_info(['overall_score', attacked['overall_score']])
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

    def get_adv(
        self, 
        sentence, sen_detect={}, ground_truth_output=0, 
        sep_size=None, step_size=None, attack_times=3,
        rept_times=10, rept_thr=0.7,
    ):
        if sep_size is None:
            sep_size=self.sep_size
        if step_size is None:
            step_size=sep_size
        for idx in range(attack_times):
            attacked=self.attack.step_attack(
                sentence, ground_truth_output, 
                window_size=sep_size, step_size=step_size,
                rept_times=rept_times, rept_thr=rept_thr
            )
            overall_score=round(np.mean(attacked['overall_score']),4)
            simi_score=round(np.mean(attacked['score']),4)
            num_queries=round(np.mean(attacked['num_queries']),3)
            budget=np.sum(attacked['budget'])
            attacked_rlt=self.wm_detector(attacked['text'])
            if attacked_rlt['is_watermarked']==False:
                break
            elif attack_times>1:
                self.log_info(['Failed', idx, 'simi_score', attacked['score']])
                self.log_info(['Failed', idx, 'adv_detect',attacked_rlt])


        self.log_info(['adv_text:', attacked['text'].replace('\n',' ')])
        self.log_info(['overall_score', overall_score])
        self.log_info(['simi_score', attacked['score']])
        self.log_info(['num_queries', num_queries])
        self.log_info(['budget', budget])
        self.log_info(['adv_detect',attacked_rlt])
        self.log_info(['wm_score drop', round(sen_detect['score']-attacked_rlt['score'], 4)])
        self.log_info(['wm_score drop rate', round((sen_detect['score']-attacked_rlt['score'])/sen_detect['score'], 4)])
        self.log_info()

        self.result_list.append({
            'raw_text': sentence,
            'raw_detect': sen_detect, #(sen_detect['is_watermarked'], round(sen_detect['score'],4)),
            'adv_text': attacked['text'],
            'overall_score': attacked['overall_score'],
            'simi_score': attacked['score'],
            'num_queries': attacked['num_queries'],
            'budget': attacked['budget'],
            'adv_detect': attacked_rlt, #(attacked_rlt['is_watermarked'], round(attacked_rlt['score'],4))
        })
        return attacked_rlt, attacked['overall_score'], num_queries, budget