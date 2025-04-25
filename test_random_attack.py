
# import torch
# import json
# from MarkLLM.watermark.auto_watermark import AutoWatermark
# from MarkLLM.utils.transformers_config import TransformersConfig
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import transformers

# import textattack
# from read_data import c4
# import textattack.attack_sems
# import datetime
import numpy as np
from textattack.utils import to_string, load_json, save_json
from llm_wm import LLM_WM
from random_attack import RandomAttack, rouge_f1, belu_func
import argparse
import Levenshtein
from defence_homo import defence_method

def test_rand_attack(
    llm_name, wm_name, max_edit_rate, max_token_num=80, atk_style='char',
    ref_tokenizer = None, ref_model=None, atk_times=1, ori_flag=False, def_stl='',
    device=0, char_op=2,
):
    
    dataset_name='../../dataset/c4/realnewslike'
    wm_data=load_json("saved_data/"+"_".join([wm_name, dataset_name.replace('/','_'), llm_name.replace('/','_')])+"_5000.json")

    device="cuda:"+str(device)
    wm_scheme=LLM_WM(model_name = llm_name, device = device, wm_name=wm_name)
    
    # if 'ZWJ' in atk_style:
    #     char_op=3
    #     atk_style='char'
    # else:
    #     char_op=2

    if ref_tokenizer is None:
        rand_attack=RandomAttack(
            wm_name=wm_name,
            tokenizer=wm_scheme.transformers_config.tokenizer,
            char_op=char_op,
        )
    else:
        rand_attack=RandomAttack(
            wm_name=wm_name,
            tokenizer=ref_tokenizer,
            ref_model=ref_model,
            ori_flag=ori_flag,
            wm_detector = wm_scheme.detect_wm,
            device = device,
            char_op=char_op,
            ppl_checker=wm_scheme.get_perplexity,
            def_stl=def_stl
        )
        rand_attack.log_info(['ref_tokenizer:', ref_tokenizer])
        rand_attack.log_info(['ref_model:', ref_model])
    
    rand_attack.log_info(['device:', device])
    rand_attack.log_info(['wm_name:', wm_name])
    rand_attack.log_info(['llm_name:', llm_name])
    rand_attack.log_info(['dataset_name:', dataset_name])
    rand_attack.log_info(['max_edit_rate:', max_edit_rate])
    rand_attack.log_info(['max_token_num:', max_token_num])
    rand_attack.log_info(['atk_style:', atk_style])
    rand_attack.log_info(['char_op:', char_op])
    rand_attack.log_info(['atk_times:', atk_times])
    rand_attack.log_info(['ori_flag:', ori_flag])
    rand_attack.log_info(['def_stl:', def_stl])
    
    count_num=0
    base_num=0
    t_edit_dist_l=[]
    c_edit_dist_l=[]
    
    target_class=0
    token_num_l=[]
    char_num_l=[]
    wm_score_l=[]
    wm_score_drop_rate_l=[]
    belu_score_l=[]
    rouge_score_l=[]
    ppl_l=[]
    adv_ppl_l=[]
    text_num=500

    adv_ocr_num=0
    wm_ocr_num=0
    adv_ocr_rate_l=[]
    wm_ocr_rate_l=[]
    ocr_adv_belu_l=[]
    ocr_wm_belu_l=[]
    ocr_adv_rouge_l=[]
    ocr_wm_rouge_l=[]
    ocr_adv_ppl_l=[]
    ocr_wm_ppl_l=[]
    ref_l=[]

    data_records=[]

    if len(atk_style)>5 or len(def_stl)>0 or atk_times>100:
        text_num=300
    for idx in range(min(int(text_num*3)+1, len(wm_data))):
        if (idx%25==0 and idx>0) or (idx>=text_num and base_num>=text_num*0.8):
            rand_attack.log_info('******')
            rand_attack.log_info({
                'edit_dist': round(np.mean(t_edit_dist_l),4),
                'token_num': round(np.mean(token_num_l),4),
                'wm_score_drop': round(np.mean(wm_score_l),4),
                'count': (count_num, base_num),
                'ref_score': round(np.mean(ref_l),4),
            })
            rand_attack.log_info({
                'wm_drop_rate': round(np.mean(wm_score_drop_rate_l),4),
                'ASR': round(count_num/base_num, 4),
                'token_budget_rate': round(np.mean(t_edit_dist_l)/np.mean(token_num_l),4),
                'char_budget_rate': round(np.mean(c_edit_dist_l)/np.mean(char_num_l),4),
                'belu': round(np.mean(belu_score_l),4),
                'rouge-f1': round(np.mean(rouge_score_l),4),
                'ppl_rate': round(np.mean(ppl_l),4),
                'adv_ppl': round(np.mean(adv_ppl_l),4),
            })
            if def_stl!='':
                rand_attack.log_info({
                    'adv_ocr_score_rate': round(np.mean(adv_ocr_rate_l),4),
                    'adv_ocr_rate': round(adv_ocr_num/base_num, 4),
                    'ocr_adv_belu': round(np.mean(ocr_adv_belu_l),4),
                    'ocr_adv_rouge': round(np.mean(ocr_adv_rouge_l),4),
                    'ocr_adv_ppl': round(np.mean(ocr_adv_ppl_l),4),
                    'wm_ocr_score_rate': round(np.mean(wm_ocr_rate_l),4),
                    'wm_ocr_rate': round(wm_ocr_num/base_num, 4),
                    'ocr_wm_belu': round(np.mean(ocr_wm_belu_l),4),
                    'ocr_wm_rouge': round(np.mean(ocr_wm_rouge_l),4),
                    'ocr_wm_ppl': round(np.mean(ocr_wm_ppl_l),4),
                })
            rand_attack.log_info('******')
        if idx>=text_num and base_num>=text_num*0.8:
            break
        
        rand_attack.log_info(str(idx))
        
        wm_text=wm_data[idx]['wm_text']
        wm_text, token_num=rand_attack.truncation(wm_text, max_token_num=max_token_num)
        if len(wm_text)==0:
            continue

        wm_det=wm_scheme.detect_wm(wm_text)
        # if wm_name == 'KGW' and wm_det['score']<6:
        #     continue
        if wm_det['is_watermarked']==True:
            base_num+=1
        else:
            continue

        adv_rlt=rand_attack.get_adv(
            wm_text, max_edit_rate=max_edit_rate,
            atk_style=atk_style,
            target_class=target_class,
            atk_times=atk_times,
        )

        adv_det=wm_scheme.detect_wm(adv_rlt['sentence'])
        if atk_times>0:
            ori_score=rand_attack.ref_score(wm_text, target_class)
            rand_attack.log_info(['ori_score:', ori_score[0]])
            rand_attack.log_info(['ref_score:', adv_rlt['ref_score']])
            ref_l.append((ori_score[0]-adv_rlt['ref_score'])/(ori_score[0]+1.5))
        rand_attack.log_info(['token_num:', token_num])
        rand_attack.log_info(['wm_detect:', wm_det])
        rand_attack.log_info(['ak_detect:', adv_det])
        rand_attack.log_info(['wm_text:', wm_text.replace('\n',' ')])
        rand_attack.log_info(['ak_text:', adv_rlt['sentence'].replace('\n',' ')])

        t_edit_dist_l.append(adv_rlt['edit_dist'])
        token_num_l.append(token_num)
        wm_score_l.append(wm_det['score']-adv_det['score'])
        char_num_l.append(len(wm_text))
        wm_score_drop_rate_l.append((wm_det['score']-adv_det['score'])/wm_det['score'])

        rouge_score=rouge_f1(wm_text, adv_rlt['sentence'])
        belu_score=belu_func(wm_text, adv_rlt['sentence'])
        c_edit_dist=Levenshtein.distance(wm_text, adv_rlt['sentence'])
        wm_ppl=wm_scheme.get_perplexity(wm_text)
        adv_ppl=wm_scheme.get_perplexity(adv_rlt['sentence'])
        rouge_score_l.append(rouge_score)
        belu_score_l.append(belu_score)
        c_edit_dist_l.append(c_edit_dist)
        ppl_l.append((adv_ppl-wm_ppl)/wm_ppl)
        adv_ppl_l.append(adv_ppl)
        
        data_record={
            'wm_text': wm_text,
            'token_num': token_num,
            'char_num': len(wm_text),
            'wm_detect': wm_det,
            'wm_ref_score': float(ori_score[0]),
            'wm_ppl': wm_ppl,
            'adv_text': adv_rlt['sentence'],
            'adv_detect': adv_det,
            'adv_ppl': adv_ppl,
            'adv_ref_score': float(adv_rlt['ref_score']),
            'wm_score_drop': wm_det['score']-adv_det['score'],
            'rouge-f1': rouge_score,
            'belu': belu_score,
            't_edit_dist': adv_rlt['edit_dist'],
            'c_edit_dist': c_edit_dist,
            'ppl_rate': (adv_ppl-wm_ppl)/wm_ppl,
        }

        if adv_det['is_watermarked']==False:
            count_num+=1

        if def_stl!='':
            ocr_adv_text=defence_method[def_stl](adv_rlt['sentence'])#, img_path='text.png'
            ocr_adv_rlt=wm_scheme.detect_wm(ocr_adv_text)
            ocr_adv_ppl=wm_scheme.get_perplexity(ocr_adv_text)
            rand_attack.log_info(['ocr_text:', ocr_adv_text.replace('\n',' ')])
            rand_attack.log_info(['ocr_detect:', ocr_adv_rlt])
            if ocr_adv_rlt['is_watermarked']==False:# and adv_det['is_watermarked']==False:
                adv_ocr_num+=1
            adv_ocr_rate_l.append((ocr_adv_rlt['score']-adv_det['score'])/(adv_det['score']+1e-4))
            ocr_adv_belu_l.append(belu_func(wm_text, ocr_adv_text))
            ocr_adv_rouge_l.append(rouge_f1(wm_text, ocr_adv_text))
            ocr_adv_ppl_l.append((ocr_adv_ppl-wm_ppl)/wm_ppl)
            
            ocr_wm_text=defence_method[def_stl](wm_text)
            ocr_wm_rlt=wm_scheme.detect_wm(ocr_wm_text)
            rand_attack.log_info(['ocr_wm_text:', ocr_wm_text.replace('\n',' ')])
            rand_attack.log_info(['ocr_wm_detect:', ocr_wm_rlt])
            if ocr_wm_rlt['is_watermarked']==False:
                wm_ocr_num+=1
            wm_ocr_rate_l.append((wm_det['score']-ocr_wm_rlt['score'])/wm_det['score'])
            ocr_wm_rouge_l.append(rouge_f1(wm_text, ocr_wm_text))
            ocr_wm_belu_l.append(belu_func(wm_text, ocr_wm_text))
            ocr_wm_ppl=wm_scheme.get_perplexity(ocr_wm_text)
            ocr_wm_ppl_l.append((ocr_wm_ppl-wm_ppl)/wm_ppl)
            data_record['ocr_adv_text']=ocr_adv_text
            data_record['ocr_adv_detect']=ocr_adv_rlt
            data_record['ocr_adv_belu']=ocr_adv_belu_l[-1]
            data_record['ocr_adv_rouge']=ocr_adv_rouge_l[-1]
            data_record['ocr_adv_ppl']=ocr_adv_ppl
            data_record['ocr_wm_text']=ocr_wm_text
            data_record['ocr_wm_detect']=ocr_wm_rlt
            data_record['ocr_wm_belu']=ocr_wm_belu_l[-1]
            data_record['ocr_wm_rouge']=ocr_wm_rouge_l[-1]
            data_record['ocr_wm_ppl']=ocr_wm_ppl
        data_records.append(data_record)
    
    attk_name='Rand'
    if char_op!=2 and 'char' in atk_style:
        attk_name='RandChar_'+str(char_op)
    save_json(
        data_records,
        "saved_attk_data/"+"_".join([
            attk_name, 
            # llm_name.replace('/','_'), wm_name.replace('/','_'), 
            str(max_edit_rate), str(max_token_num), atk_style, 
            str(atk_times), str(ori_flag), def_stl, 
            ref_model.replace('saved_model/',''), 
        ])+".json"
    )

if __name__=="__main__":
    # python test_random_attack.py --max_edit_rate 0.2 --atk_style "char" --max_token_num 200
    parser = argparse.ArgumentParser(description='test_rand_attack')
    parser.add_argument('--llm_name', type=str, default='../model/Llama3.1-8B_hg')
    parser.add_argument('--wm_name', type=str, default='KGW')
    parser.add_argument('--max_edit_rate', type=float, default=0.5)
    parser.add_argument('--max_token_num', type=int, default=100)
    parser.add_argument('--atk_style', type=str, default='low')

    parser.add_argument('--ref_tokenizer', type=str, default='bert-base-uncased')#'bert-base-uncased'
    parser.add_argument('--ref_model', type=str)#, default='saved_model/RefDetector_Unigram_.._.._dataset_c4_realnewslike_facebook_opt-1.3b_bert-base-uncased_2025-01-14')
    parser.add_argument('--atk_times', type=int, default=0)
    parser.add_argument('--ori_flag', type=str, default='False')
    parser.add_argument('--def_stl', type=str, default='')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--char_op', type=int, default=2)
    
    args = parser.parse_args()
    test_rand_attack(
        llm_name=args.llm_name,
        wm_name=args.wm_name, 
        max_edit_rate=args.max_edit_rate,
        max_token_num=args.max_token_num,
        atk_style=args.atk_style,
        ref_tokenizer=args.ref_tokenizer,
        ref_model=args.ref_model,
        atk_times=args.atk_times,
        ori_flag=bool(args.ori_flag=='True'),
        def_stl=args.def_stl,
        device=args.device,
        char_op=args.char_op,
    )
    