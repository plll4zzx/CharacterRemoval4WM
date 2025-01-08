
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
import argparse


def test_rand_attack(
    wm_name, max_edit_rate, max_token_num=80, atk_style='char',
    ref_tokenizer = None, ref_model=None, atk_times=1
):
    
    llm_name="facebook/opt-1.3b"
    dataset_name='../../dataset/c4/realnewslike'
    wm_data=load_json("saved_data/"+"_".join([wm_name, dataset_name.replace('/','_'), llm_name.replace('/','_')])+"_5000.json")

    wm_scheme=LLM_WM(model_name = llm_name, device = "cuda", wm_name=wm_name)
    
    if ref_tokenizer is None:
        rand_attack=RandomAttack(
            tokenizer=wm_scheme.transformers_config.tokenizer,
        )
    else:
        rand_attack=RandomAttack(
            tokenizer=ref_tokenizer,
            ref_model=ref_model,
        )
        rand_attack.log_info(['ref_tokenizer:', ref_tokenizer])
        rand_attack.log_info(['ref_model:', ref_model])
    
    rand_attack.log_info(['wm_name:', wm_name])
    rand_attack.log_info(['llm_name:', llm_name])
    rand_attack.log_info(['dataset_name:', dataset_name])
    rand_attack.log_info(['max_edit_rate:', max_edit_rate])
    rand_attack.log_info(['max_token_num:', max_token_num])
    rand_attack.log_info(['atk_style:', atk_style])
    rand_attack.log_info(['atk_times:', atk_times])
    
    count_num=0
    base_num=0
    edit_dist_l=[]
    
    target_class=0
    token_num_l=[]
    wm_score_l=[]
    wm_score_drop_rate_l=[]
    text_num=300
    for idx in range(text_num+1):
        
        rand_attack.log_info(str(idx))
        if idx%25==0 and idx>0:
            rand_attack.log_info('******')
            rand_attack.log_info(['ASR', round(count_num/base_num,4)])
            rand_attack.log_info(['edit_dist', round(np.mean(edit_dist_l),4)])
            rand_attack.log_info(['token_num', round(np.mean(token_num_l),4)])
            rand_attack.log_info(['budget rate', round(np.mean(edit_dist_l)/np.mean(token_num_l),4)])
            rand_attack.log_info(['wm_score drop', round(np.mean(wm_score_l),3)])
            rand_attack.log_info(['wm_score drop rate', round(np.mean(wm_score_drop_rate_l),4)])
            rand_attack.log_info('******')
            if idx==text_num:
                break
        
        wm_text=wm_data[idx]['wm_text']
        wm_text, token_num=rand_attack.truncation(wm_text, max_token_num=max_token_num)
        if len(wm_text)==0:
            continue

        wm_rlt=wm_scheme.detect_wm(wm_text)
        if wm_rlt['is_watermarked']==True:
            base_num+=1
        else:
            continue

        adv_rlt=rand_attack.get_adv(
            wm_text, max_edit_rate=max_edit_rate,
            atk_style=atk_style,
            target_class=target_class,
            atk_times=atk_times,
        )

        attk_rlt=wm_scheme.detect_wm(adv_rlt['sentence'])
        
        ori_score=rand_attack.ref_score(wm_text, target_class)
        rand_attack.log_info(['ori_score:', ori_score[0]])
        rand_attack.log_info(['ref_score:', adv_rlt['ref_score']])
        rand_attack.log_info(['wm_detect:', wm_rlt])
        rand_attack.log_info(['ak_detect:', attk_rlt])
        rand_attack.log_info(['wm_text:', wm_text.replace('\n',' ')])
        rand_attack.log_info(['ak_text:', adv_rlt['sentence'].replace('\n',' ')])

        edit_dist_l.append(adv_rlt['edit_dist'])
        token_num_l.append(token_num)
        wm_score_l.append(wm_rlt['score']-attk_rlt['score'])
        wm_score_drop_rate_l.append((wm_rlt['score']-attk_rlt['score'])/wm_rlt['score'])

        if attk_rlt['is_watermarked']==False:
            count_num+=1
    # rand_attack.save()

if __name__=="__main__":
    # python test_random_attack.py --max_edit_rate 0.2 --atk_style "char" --max_token_num 200
    parser = argparse.ArgumentParser(description='test_rand_attack')
    parser.add_argument('--wm_name', type=str, default='KGW')
    parser.add_argument('--max_edit_rate', type=float, default=0.05)
    parser.add_argument('--max_token_num', type=int, default=100)
    parser.add_argument('--atk_style', type=str, default='char')

    parser.add_argument('--ref_tokenizer', type=str, default='bert-base-uncased')
    parser.add_argument('--ref_model', type=str, default='saved_model/RefDetector_KGW_.._.._dataset_c4_realnewslike_facebook_opt-1.3b_2025-01-05')
    parser.add_argument('--atk_times', type=int, default=50)
    
    args = parser.parse_args()
    test_rand_attack(
        wm_name=args.wm_name, 
        max_edit_rate=args.max_edit_rate,
        max_token_num=args.max_token_num,
        atk_style=args.atk_style,
        ref_tokenizer=args.ref_tokenizer,
        ref_model=args.ref_model,
        atk_times=args.atk_times
    )
    