
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
from ref_attack import RefAttack
import argparse

def test_sam_attack(
    wm_name, 
    # query_budget, slide_flag, sep_size=30, edit_distance=5, 
    max_token_num=80
):
    llm_name="facebook/opt-1.3b"
    dataset_name='../../dataset/c4/realnewslike'

    wm_data=load_json("saved_data/"+"_".join([wm_name, dataset_name.replace('/','_'), llm_name.replace('/','_')])+"_5000.json")

    wm_scheme=LLM_WM(model_name = llm_name, device = "cuda", wm_name=wm_name)
    
    ref_attack=RefAttack(
        # sep_size=sep_size,
        attack_name = 'TextBuggerCharLi2018',
        victim_model = 'saved_model/RefDetector_KGW_.._.._dataset_c4_realnewslike_facebook_opt-1.3b_2024-12-31',
        victim_tokenizer = 'bert-base-uncased',
        llm_name=llm_name,
        wm_name=wm_name,
        wm_detector=wm_scheme.detect_wm,
    )
    ref_attack.log_info(['max_token_num:', max_token_num])
    
    count_num=0
    base_num=0
    cls_score_l=[]
    num_queries_l=[]
    budget_l=[]
    token_num_l=[]
    wm_score_l=[]
    wm_score_drop_rate_l=[]
    ref_acc_l=[]
    for idx in range(100):#len(wm_data)
        
        wm_text=wm_data[idx]['wm_text']
        wm_text, token_num=ref_attack.truncation(wm_text, max_token_num=max_token_num)
        if len(wm_text)==0:
            continue

        wm_rlt=wm_scheme.detect_wm(wm_text)
        ref_attack.log_info(str(idx))
        if wm_rlt['is_watermarked']==True:
            base_num+=1
        else:
            continue
        ref_attack.log_info(['wm_text:', wm_text.replace('\n',' ')])
        ref_attack.log_info(['wm_detect:', wm_rlt])
        ref_attack.log_info(['token_num:', token_num])

        # token_importance=ref_attack.get_gradient(wm_text, ground_truth_output=0)

        # ref_acc=ref_attack.gradient_match(
        #     wm_text, token_importance, wm_rlt['green_token_flags'], 
        #     wm_tokenizer=wm_scheme.wm_model.config.generation_tokenizer, 
        #     ref_tokenizer=ref_attack.tokenizer
        # )
        # ref_acc_l.append(ref_acc[0])
        # ref_attack.log_info(['ref_acc', ref_acc])

        attk_rlt, cls_score, num_queries, budget=ref_attack.get_adv(
            wm_text, wm_rlt, ground_truth_output=0, 
            # sep_size=sep_size, 
            attack_times=1,
            rept_times=1, rept_thr=0.8
        )

        cls_score_l.append(cls_score)
        num_queries_l.append(num_queries)
        budget_l.append(budget)
        token_num_l.append(token_num)
        wm_score_l.append(wm_rlt['score']-attk_rlt['score'])
        wm_score_drop_rate_l.append((wm_rlt['score']-attk_rlt['score'])/wm_rlt['score'])

        if attk_rlt['is_watermarked']==False:
            count_num+=1
        
        if base_num%25==0 and base_num>0:
            ref_attack.log_info('******')
            ref_attack.log_info(['ASR', round(count_num/base_num,4)])
            ref_attack.log_info(['cls_score', round(np.mean(cls_score_l),4)])
            ref_attack.log_info(['num_queries', round(np.mean(num_queries_l),3)])
            ref_attack.log_info(['budget', round(np.mean(budget_l),3)])
            ref_attack.log_info(['token_num', round(np.mean(token_num_l),3)])
            ref_attack.log_info(['budget rate', round(np.mean(budget_l)/np.mean(token_num_l),4)])
            ref_attack.log_info(['wm_score drop', round(np.mean(wm_score_l),3)])
            ref_attack.log_info(['wm_score drop rate', round(np.mean(wm_score_drop_rate_l),4)])
            ref_attack.log_info('******')
    
    ref_attack.log_info('******')
    ref_attack.log_info(['ASR', round(count_num/base_num,4)])
    ref_attack.log_info(['cls_score', round(np.mean(cls_score_l),4)])
    ref_attack.log_info(['num_queries', round(np.mean(num_queries_l),3)])
    ref_attack.log_info(['budget', round(np.mean(budget_l),3)])
    ref_attack.log_info(['token_num', round(np.mean(token_num_l),3)])
    ref_attack.log_info(['budget rate', round(np.mean(budget_l)/np.mean(token_num_l),4)])
    ref_attack.log_info(['wm_score drop', round(np.mean(wm_score_l),3)])
    ref_attack.log_info(['wm_score drop rate', round(np.mean(wm_score_drop_rate_l),4)])
    # ref_attack.log_info(['ref_acc_l', round(np.mean(ref_acc_l),4)])
    ref_attack.log_info('******')
    ref_attack.save()

if __name__=="__main__":
    
    # wm_name='TS'
    # query_budget=500
    # slide_flag=True
    parser = argparse.ArgumentParser(description='test_ref_attack')
    parser.add_argument('--wm_name', type=str, default='KGW')
    # parser.add_argument('--query_budget', type=int, default=500)
    # parser.add_argument('--slide_flag', type=str, default='True')
    # parser.add_argument('--sep_size', type=int, default=12)
    # parser.add_argument('--edit_distance', type=int, default=5)
    parser.add_argument('--max_token_num', type=int, default=200)
    
    args = parser.parse_args()
    test_sam_attack(
        wm_name=args.wm_name, 
        # query_budget=args.query_budget, 
        # slide_flag=args.slide_flag=='True',
        # sep_size=args.sep_size,
        # edit_distance=args.edit_distance,
        max_token_num=args.max_token_num
    )