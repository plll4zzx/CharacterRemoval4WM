

from textattack.utils import load_json, save_json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, BertForSequenceClassification
from torch.optim import AdamW, SGD, Adam
from torch.utils.data import Dataset, DataLoader
import torch
from accelerate.test_utils.testing import get_backend
from tqdm import tqdm
import evaluate
import numpy as np
import os
import datetime
import random
from llm_wm import LLM_WM

class TokenizerAttack:
    def __init__(self, tokenizer_path=None, text_len=None, wm_detector=None):
        
        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer_path)
        self.wm_detector=wm_detector
        self.text_len=text_len
        

    def attack(self, text):
        tmp_ids=self.tokenizer.encode(text, add_special_tokens=False)[0:self.text_len]
        new_text=self.tokenizer.decode(tmp_ids, skip_special_tokens=True)
        wm_result=self.wm_detector(new_text)
        return new_text, wm_result

if __name__=='__main__':

    tokenizer_path='google-t5/t5-base'#'../model/llama-2-7b'# "bert-base-uncased" # openai-community/gpt2, FacebookAI/roberta-base google-t5/t5-base
    llm_name="facebook/opt-1.3b"
    dataset_name='../../dataset/c4/realnewslike'
    wm_name='TS'
    data_num=5000
    
    for wm_name in ['DIP','Unbiased','SynthID']:#,'KGW','TS','SIR'
        data_path="saved_data/"+"_".join([wm_name, dataset_name.replace('/','_'), llm_name.replace('/','_')])+"_5000.json"
        dataset=load_json(data_path)[0:data_num]
        llm_wm=LLM_WM(model_name = llm_name, device = "cuda", wm_name=wm_name)
        t_attack=TokenizerAttack(
            tokenizer_path=tokenizer_path, 
            text_len=200, 
            wm_detector=llm_wm.detect_wm
        )
        
        count_un=0
        count_wm=0
        new_dataset=[]
        for tmp_d in tqdm(dataset, ncols=100):
            if tmp_d['wm_detect']['is_watermarked']==True:
                tmp_text=tmp_d['wm_text']
                tmp_labels=1
                new_text, wm_result = t_attack.attack(tmp_text)
                if wm_result['is_watermarked']==False:
                    count_un+=1
                else:
                    count_wm+=1
            new_dataset.append({
                'raw_text':tmp_text,
                'new_text':new_text,
                'wm_result': wm_result,
            })
                    
        print(wm_name, count_wm, count_un)
        save_json(new_dataset, data_path[0:-5]+'_'+tokenizer_path.replace('/','_')+'.json')