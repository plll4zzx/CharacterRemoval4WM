

import numpy as np
from textattack.utils import to_string, load_json, save_json, compute_auc, compute_log_auc, truncation
from llm_wm import LLM_WM
from random_attack import RandomAttack, rouge_f1, belu_func
import argparse
import Levenshtein
from defence_homo import defence_method
import os

def detect_un_text(
    llm_name, wm_name, 
    max_token_num=80, 
    device=0, 
):
    
    dataset_name='../../dataset/c4/realnewslike'
    wm_data=load_json("saved_data/"+"_".join([wm_name, dataset_name.replace('/','_'), llm_name.replace('/','_')])+"_5000.json")

    device="cuda:"+str(device)
    wm_scheme=LLM_WM(model_name = llm_name, device = device, wm_name=wm_name)
    
    text_num=500

    data_records=[]

    for idx in range(min(int(text_num*3)+1, len(wm_data))):
        
        if idx>=text_num:
            break
        
        un_text=wm_data[idx]['un_text']
        un_text, un_token_num=truncation(un_text, max_token_num=max_token_num)
        if len(un_text)==0:
            continue

        un_det=wm_scheme.detect_wm(un_text)
        
        data_record={
            'un_text': un_text,
            'token_num': un_token_num,
            'char_num': len(un_text),
            'un_detect': un_det,
        }
        data_records.append(data_record)
    save_json(
        data_records,
        "saved_attk_data/"+"_".join([
            'un_text',
            llm_name.replace('/','_'), wm_name.replace('/','_'), 
            str(max_token_num), 
        ])+".json"
    )

if __name__=="__main__":

    for llm_name in ['facebook/opt-1.3b']:#'../model/Llama3.1-8B_hg', 
        for wm_name in ['KGW','DIP', 'SynthID','Unigram','Unbiased']:
            for max_token_num in [50, 100, 150, 200, 250]:
                detect_un_text(
                    llm_name, wm_name, 
                    max_token_num=max_token_num, 
                    device=0, 
                )
    