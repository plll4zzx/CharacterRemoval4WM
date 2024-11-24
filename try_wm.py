
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
from textattack.utils import Logger
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
    
class SemanticAttack:

    def __init__(
        self,
        target_cos=0.3,
        edit_distance=3,
        query_budget=500,
        attack_name = 'TextBuggerLi2018',
        victim_name = 'sentence-transformers/all-distilroberta-v1',#'sentence-transformers/all-mpnet-base-v2'#
        logger=None,
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

    def log_info(self, info):
        self.log.logger.info(info)



if __name__=="__main__":
    
    file_num=10
    file_data_num=100
    dataset_name='../../dataset/c4/realnewslike'
    file_num=int(file_num)


    text_len=50
    c4_dataset=c4(dir_path=dataset_name, file_num=file_num, file_data_num=file_data_num)
    c4_dataset.load_data(text_len)

    wm_scheme=LLM_WM(model_name = "facebook/opt-1.3b", device = "cuda", wm_name='SIR')
    
    target_cos=0.3
    edit_distance=3
    query_budget=500
    attack_name = 'TextBuggerLi2018'
    victim_name = 'sentence-transformers/all-distilroberta-v1'#'sentence-transformers/all-mpnet-base-v2'#
    model = transformers.AutoModel.from_pretrained(victim_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(victim_name)
    model_wrapper = textattack.models.wrappers.HuggingFaceEncoderWrapper(model, tokenizer)
    attack = getattr(textattack.attack_sems, attack_name).build(
        model_wrapper, 
        target_cos=target_cos, 
        edit_distance=edit_distance, 
        query_budget=query_budget
    )
    
    count_num=0
    base_num=0
    simi_score_l=[]
    num_queries_l=[]
    budget_l=[]
    for idx in range(0,200,1):
        wm_text, un_wm_text = wm_scheme.generate(c4_dataset.data[1+idx][0][0:500], wm_seed=123)
        wm_text=wm_text[0:500]
        un_wm_text=un_wm_text[0:500]

        wm_rlt=wm_scheme.detect_wm(wm_text)
        # un_wm_rlt=wm_scheme.detect_wm(un_wm_text)
        print(idx)
        if wm_rlt['is_watermarked']==True:# and un_wm_rlt['is_watermarked']==False:
            base_num+=1
        else:
            continue
        print('WM_TEXT:', wm_text.replace('\n',' '))
        print('wm', wm_rlt)
        # print('un', un_wm_rlt)

        attacked=attack.step_attack(wm_text, 0, window_size=40, step_ize=40)
        simi_score=round(np.mean(attacked['score']),4)
        num_queries=round(np.mean(attacked['num_queries']),3)
        budget=np.sum(attacked['budget'])
        print('ATTACKED:', attacked['text'].replace('\n',' '))
        print('simi_score', simi_score, '; num_queries', num_queries, '; budget', budget)
        attacked_rlt=wm_scheme.detect_wm(attacked['text'])
        print('attacked',attacked_rlt)
        print()


        simi_score_l.append(simi_score)
        num_queries_l.append(num_queries)
        budget_l.append(budget)

        if attacked_rlt['is_watermarked']==False:
            count_num+=1
        
        if base_num%25==0 and base_num>0:
            print(count_num, base_num)
            print('simi_score', round(np.mean(simi_score_l),4))
            print('num_queries', round(np.mean(num_queries_l),3))
            print('budget', round(np.mean(budget_l),3))