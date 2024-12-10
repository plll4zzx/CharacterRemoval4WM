
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

    wm_data=load_json("saved_data/SemStamp_.._.._dataset_c4_realnewslike_facebook_opt-1.3b_200.json")

    wm_scheme=LLM_WM(model_name = "facebook/opt-1.3b", device = "cuda", wm_name='SemStamp')
    
    rand_attack=RandomAttack(
        tokenizer=wm_scheme.transformers_config.tokenizer,

    )
    max_edit_rate=0.5
    rand_attack.log_info(['max_edit_rate:', max_edit_rate])
    
    count_num=0
    base_num=0
    token_num=120
    edit_dist_l=[]
    num_queries_l=[]
    budget_l=[]
    for idx in range(len(wm_data)):
        
        wm_text=wm_data[idx]['wm_text']
        wm_text=rand_attack.truncation(wm_text, token_num=token_num)
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
        # un_rlt=wm_scheme.detect_wm(un_wm_text)
        # sem_attack.log_info(['un_detect:', un_rlt])

        attk_text, edit_dist=rand_attack.wm_wipe(
            wm_text, max_edit_rate=max_edit_rate
        )

        att_rlt=wm_scheme.detect_wm(attk_text)
        rand_attack.log_info(['attk_detect:', att_rlt])
        edit_dist_l.append(edit_dist)
        # num_queries_l.append(num_queries)
        # budget_l.append(budget)

        if att_rlt['is_watermarked']==False:
            count_num+=1
        
        if base_num%25==0 and base_num>0:
            rand_attack.log_info([count_num, base_num])
            rand_attack.log_info(['edit_dist', round(np.mean(edit_dist_l),4)])
    #         rand_attack.log_info(['num_queries', round(np.mean(num_queries_l),3)])
    #         rand_attack.log_info(['budget', round(np.mean(budget_l),3)])
    
    rand_attack.log_info([count_num, base_num])
    rand_attack.log_info(['edit_dist', round(np.mean(edit_dist_l),4)])
    # rand_attack.log_info(['num_queries', round(np.mean(num_queries_l),3)])
    # rand_attack.log_info(['budget', round(np.mean(budget_l),3)])
    # rand_attack.save()