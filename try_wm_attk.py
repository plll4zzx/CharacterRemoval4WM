
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
from semantic_attack import SemanticAttack


if __name__=="__main__":
    
    # file_num=10
    # file_data_num=100
    # dataset_name='../../dataset/c4/realnewslike'
    # file_num=int(file_num)

    # text_len=50
    # c4_dataset=c4(dir_path=dataset_name, file_num=file_num, file_data_num=file_data_num)
    # c4_dataset.load_data(text_len)

    # wm_data=load_json("saved_data/SemStamp_.._.._dataset_c4_realnewslike_facebook_opt-1.3b_200.json")
    wm_data=load_json("saved_data/SemStamp_.._.._dataset_c4_realnewslike_facebook_opt-1.3b_200.json")

    wm_scheme=LLM_WM(model_name = "facebook/opt-1.3b", device = "cuda", wm_name='SemStamp')
    
    sem_attack=SemanticAttack(
        target_cos=0.3,
        edit_distance=5,
        query_budget=500,
        temperature=50,
        random_num=5, 
        random_one=True,
        attack_name = 'TextBuggerLi2018',
        victim_name = 'sentence-transformers/all-distilroberta-v1',
        llm_name="facebook/opt-1.3b",
        wm_name='SemStamp',
        wm_detector=wm_scheme.detect_wm,
        max_single_query=100,
    )
    
    max_token_num=120
    count_num=0
    base_num=0
    simi_score_l=[]
    num_queries_l=[]
    budget_l=[]
    token_num_l=[]
    for idx in range(2,len(wm_data)):
        # wm_text, un_wm_text = wm_scheme.generate(
        #     c4_dataset.data[1+idx][0][0:500], 
        #     wm_seed=123, 
        #     # un_wm_flag=True
        # )
        wm_text=wm_data[idx]['wm_text']
        wm_text, token_num=sem_attack.truncation(wm_text, max_token_num=max_token_num)
        if len(wm_text)==0:
            continue
        # un_wm_text=un_wm_text[0:500]

        wm_rlt=wm_scheme.detect_wm(wm_text)
        sem_attack.log_info(str(idx))
        if wm_rlt['is_watermarked']==True:
            base_num+=1
        else:
            continue
        sem_attack.log_info(['wm_text:', wm_text.replace('\n',' ')])
        sem_attack.log_info(['wm_detect:', wm_rlt])
        sem_attack.log_info(['token_num:', token_num])
        # un_rlt=wm_scheme.detect_wm(un_wm_text)
        # sem_attack.log_info(['un_detect:', un_rlt])

        is_watermarked, simi_score, num_queries, budget=sem_attack.get_adv(
            wm_text, wm_rlt, 1, 
            window_size=30, step_ize=30, attack_times=1,
            rept_times=1, rept_thr=0.8
        )

        simi_score_l.append(simi_score)
        num_queries_l.append(num_queries)
        budget_l.append(budget)
        token_num_l.append(token_num)

        if is_watermarked==False:
            count_num+=1
        
        if base_num%25==0 and base_num>0:
            sem_attack.log_info([count_num, base_num])
            sem_attack.log_info(['simi_score', round(np.mean(simi_score_l),4)])
            sem_attack.log_info(['num_queries', round(np.mean(num_queries_l),3)])
            sem_attack.log_info(['budget', round(np.mean(budget_l),3)])
            sem_attack.log_info(['token_num', round(np.mean(token_num_l),3)])
    
    sem_attack.log_info([count_num, base_num])
    sem_attack.log_info(['simi_score', round(np.mean(simi_score_l),4)])
    sem_attack.log_info(['num_queries', round(np.mean(num_queries_l),3)])
    sem_attack.log_info(['budget', round(np.mean(budget_l)/np.mean(token_num_l),3)])
    sem_attack.save()