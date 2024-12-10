
# import torch
# import json
# from MarkLLM.watermark.auto_watermark import AutoWatermark
# from MarkLLM.utils.transformers_config import TransformersConfig
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import transformers

# import textattack
from read_data import c4
# import textattack.attack_sems
import numpy as np
from textattack.utils import Logger, to_string, load_json
# import datetime
from llm_wm import LLM_WM
from random_attack import RandomAttack


if __name__=="__main__":
    
    # file_num=10
    # file_data_num=100
    # dataset_name='../../dataset/c4/realnewslike'
    # file_num=int(file_num)

    # text_len=50
    # c4_dataset=c4(dir_path=dataset_name, file_num=file_num, file_data_num=file_data_num)
    # c4_dataset.load_data(text_len)

    wm_name='TS'
    llm_name="facebook/opt-1.3b"
    dataset_name='../../dataset/c4/realnewslike'
    wm_data=load_json("saved_data/"+"_".join([wm_name, dataset_name.replace('/','_'), llm_name.replace('/','_')])+"_5000.json")

    wm_scheme=LLM_WM(model_name = llm_name, device = "cuda", wm_name=wm_name)
    
    rand_attack=RandomAttack(
        tokenizer=wm_scheme.transformers_config.tokenizer,
    )
    max_edit_rate=1/6
    rand_attack.log_info(['max_edit_rate:', max_edit_rate])
    
    count_num=0
    base_num=0
    max_token_num=120
    edit_dist_l=[]
    num_queries_l=[]
    budget_l=[]
    token_num_l=[]
    wm_score_l=[]
    for idx in range(200):
        
        wm_text=wm_data[idx]['wm_text']
        wm_text, token_num=rand_attack.truncation(wm_text, max_token_num=max_token_num)
        if len(wm_text)==0:
            continue

        wm_rlt=wm_scheme.detect_wm(wm_text)
        rand_attack.log_info(str(idx))
        if wm_rlt['is_watermarked']==True:
            base_num+=1
        else:
            continue
        rand_attack.log_info(['wm_text:', wm_text.replace('\n',' ')])
        rand_attack.log_info(['wm_detect:', wm_rlt])

        attk_text, edit_dist=rand_attack.wm_wipe(
            wm_text, max_edit_rate=max_edit_rate
        )

        attk_rlt=wm_scheme.detect_wm(attk_text)
        rand_attack.log_info(['attk_detect:', attk_rlt])
        edit_dist_l.append(edit_dist)
        token_num_l.append(token_num)
        wm_score_l.append(wm_rlt['score']-attk_rlt['score'])

        if attk_rlt['is_watermarked']==False:
            count_num+=1
        
        if idx%25==0 and idx>0:
            rand_attack.log_info('******')
            rand_attack.log_info(['ASR', round(count_num/base_num,4)])
            rand_attack.log_info(['edit_dist', round(np.mean(edit_dist_l),4)])
            rand_attack.log_info(['token_num', round(np.mean(token_num_l),4)])
            rand_attack.log_info(['wm_score drop', round(np.mean(wm_score_l),3)])
            rand_attack.log_info('******')
    
    rand_attack.log_info('******')
    rand_attack.log_info(['ASR', round(count_num/base_num,4)])
    rand_attack.log_info(['edit_dist', round(np.mean(edit_dist_l),4)])
    rand_attack.log_info(['token_num', round(np.mean(token_num_l),4)])
    rand_attack.log_info(['wm_score drop', round(np.mean(wm_score_l),3)])
    rand_attack.log_info('******')
    # rand_attack.save()