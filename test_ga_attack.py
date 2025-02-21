
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
from ga_attack import GA_Attack
import argparse
import Levenshtein
from random_attack import rouge_f1, belu_func


def test_ga_attack(
    wm_name, max_edit_rate, num_generations, 
    max_token_num=80, victim_tokenizer = 'bert-base-uncased',
    victim_model = 'saved_model/RefDetector_KGW_.._.._dataset_c4_realnewslike_facebook_opt-1.3b_2024-12-31',
    llm_name="facebook/opt-1.3b",
    dataset_name='../../dataset/c4/realnewslike',
    len_weight=1,
    fitness_threshold=0.9,
    eva_thr=0.2,
    mean=0,
    std=1,
    ab_std=1,
    atk_style='char',
    ori_flag=False
):
    wm_data=load_json("saved_data/"+"_".join([wm_name, dataset_name.replace('/','_'), llm_name.replace('/','_')])+"_5000.json")

    wm_scheme=LLM_WM(model_name = llm_name, device = "cuda", wm_name=wm_name)
    
    ga_attack=GA_Attack(
        victim_model = victim_model,
        victim_tokenizer = victim_tokenizer,
        wm_detector = wm_scheme.detect_wm,
        wm_name = wm_name,
        len_weight=len_weight,
        fitness_threshold=fitness_threshold,
        eva_thr=eva_thr,
        mean=mean,
        std=std,
        ab_std=ab_std,
        atk_style=atk_style,
        ori_flag=ori_flag
    )
    if ori_flag==True:
        ab_std=100
    ga_attack.log_info(['wm_name:', wm_name])
    ga_attack.log_info(['llm_name:', llm_name])
    ga_attack.log_info(['victim_tokenizer:', victim_tokenizer])
    ga_attack.log_info(['victim_model:', victim_model])
    ga_attack.log_info(['dataset_name:', dataset_name])
    ga_attack.log_info(['max_edit_rate:', max_edit_rate])
    ga_attack.log_info(['num_generations:', num_generations])
    ga_attack.log_info(['max_token_num:', max_token_num])
    ga_attack.log_info(['len_weight:', len_weight])
    ga_attack.log_info(['fitness_threshold:', fitness_threshold])
    ga_attack.log_info(['eva_thr:', eva_thr])
    ga_attack.log_info(['ab_std:', ab_std])
    ga_attack.log_info(['atk_style:', atk_style])
    ga_attack.log_info(['ori_flag:', ori_flag])
    
    target_class=0
    count_num=0
    base_num=0
    t_edit_dist_l=[]
    c_edit_dist_l=[]
    
    token_num_l=[]
    char_num_l=[]
    wm_score_l=[]
    wm_score_drop_rate_l=[]
    belu_score_l=[]
    rouge_score_l=[]
    ppl_l=[]
    adv_ppl_l=[]

    text_num=300
    for idx in range(text_num+1):#[79]:#
        if idx%25==0 and idx>0:
            ga_attack.log_info('******')
            ga_attack.log_info({
                'edit_dist': round(np.mean(t_edit_dist_l),4),
                'token_num': round(np.mean(token_num_l),4),
                'wm_score_drop': round(np.mean(wm_score_l),4),
                'count': (count_num, base_num),
            })
            ga_attack.log_info({
                'wm_drop_rate': round(np.mean(wm_score_drop_rate_l),4),
                'ASR': round(count_num/base_num, 4),
                'token_budget_rate': round(np.mean(t_edit_dist_l)/np.mean(token_num_l),4),
                'char_budget_rate': round(np.mean(c_edit_dist_l)/np.mean(char_num_l),4),
                'belu': round(np.mean(belu_score_l),4),
                'rouge-f1': round(np.mean(rouge_score_l),4),
                'ppl_rate': round(np.mean(ppl_l),4),
                'adv_ppl': round(np.mean(adv_ppl_l),4)
            })
            ga_attack.log_info('******')
        if idx==text_num:
            break
        
        ga_attack.log_info(str(idx))
        
        wm_text=wm_data[299-idx]['wm_text']
        wm_text, token_num=ga_attack.truncation(wm_text, max_token_num=max_token_num)
        if len(wm_text)==0:
            continue

        wm_rlt=wm_scheme.detect_wm(wm_text)
        if wm_rlt['is_watermarked']==True:
            base_num+=1
        else:
            continue
        
        ori_fitness=ga_attack.evaluate_fitness(wm_text, target_class)
        ga_attack.log_info(['ori_fitness:', ori_fitness])
        ga_attack.log_info(['wm_detect:', wm_rlt])

        attk_text, edit_dist, attk_score=ga_attack.get_adv(
            wm_text, target_class, ori_fitness,
            max_edit_rate=max_edit_rate,
            num_generations=num_generations,
        )
        try:
            attk_rlt=wm_scheme.detect_wm(attk_text)
            ga_attack.log_info(['ak_detect:', attk_rlt])
            wm_score_l.append(wm_rlt['score']-attk_rlt['score'])
            wm_score_drop_rate_l.append((wm_rlt['score']-attk_rlt['score'])/wm_rlt['score'])
            rouge_score_l.append(rouge_f1(wm_text, attk_text))
            belu_score_l.append(belu_func(wm_text, attk_text))
            char_num_l.append(len(wm_text))
            c_edit_dist_l.append(Levenshtein.distance(wm_text, attk_text))

            wm_ppl=wm_scheme.get_perplexity(wm_text)
            adv_ppl=wm_scheme.get_perplexity(attk_text)
            ppl_l.append((adv_ppl-wm_ppl)/wm_ppl)
            adv_ppl_l.append(adv_ppl)

            if attk_rlt['is_watermarked']==False:
                count_num+=1
        except:
            ga_attack.log_info('ERROR')
        ga_attack.log_info(['wm_text:', wm_text.replace('\n',' ')])
        ga_attack.log_info(['ak_text:', attk_text.replace('\n',' ')])
        t_edit_dist_l.append(edit_dist)
        token_num_l.append(token_num)
    
    # ga_attack.save()

if __name__=="__main__":
    # python test_ga_attack.py --num_generations 10 --max_edit_rate 0.2 --max_token_num 200 --victim_model --wm_name
    parser = argparse.ArgumentParser(description='test_ga_attack')
    parser.add_argument('--wm_name', type=str, default='KGW')
    parser.add_argument('--max_edit_rate', type=float, default=0.1)
    parser.add_argument('--len_weight', type=float, default=1.3)
    parser.add_argument('--eva_thr', type=float, default=0.1)
    parser.add_argument('--fitness_threshold', type=float, default=0.9)
    parser.add_argument('--max_token_num', type=int, default=100)
    parser.add_argument('--num_generations', type=int, default=15)
    parser.add_argument('--victim_tokenizer', type=str, default='facebook/opt-350m')
    parser.add_argument('--victim_model', type=str, default='saved_model/RefDetector_KGW_.._.._dataset_c4_realnewslike_facebook_opt-1.3b_facebook_opt-350m_2025-01-08')
    
    parser.add_argument('--llm_name', type=str, default='')
    parser.add_argument('--mean', type=float, default=0)
    parser.add_argument('--std', type=float, default=1)
    parser.add_argument('--ab_std', type=float, default=3)
    parser.add_argument('--atk_style', type=str, default='char')
    parser.add_argument('--ori_flag', type=str, default='False')
    
    args = parser.parse_args()
    test_ga_attack(
        wm_name=args.wm_name, 
        max_edit_rate=args.max_edit_rate,
        max_token_num=args.max_token_num,
        num_generations=args.num_generations,
        victim_model=args.victim_model,
        victim_tokenizer=args.victim_tokenizer,
        len_weight=args.len_weight,
        fitness_threshold=args.fitness_threshold,
        eva_thr=args.eva_thr,
        llm_name=args.llm_name, #
        mean=args.mean, #
        std=args.std, #
        ab_std=args.ab_std, #
        atk_style=args.atk_style, #
        ori_flag=args.ori_flag #
    )
    