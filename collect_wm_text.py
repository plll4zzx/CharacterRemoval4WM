
from read_data import c4
# import textattack.attack_sems
import numpy as np
from textattack.utils import Logger, to_string, save_json, get_advtext_filename
# import datetime
from llm_wm import LLM_WM
from semantic_attack import SemanticAttack
import os
from tqdm import tqdm

def get_wm_data(
    file_num=10,
    file_data_num=20,
    dataset_name='../../dataset/c4/realnewslike',
    text_len=50,
    wm_name='SIR',#'SemStamp'
    model_name = "facebook/opt-1.3b",
):
    c4_dataset=c4(dir_path=dataset_name, file_num=file_num, file_data_num=file_data_num)
    c4_dataset.load_data(text_len)

    wm_scheme=LLM_WM(model_name = model_name, device = "cuda", wm_name=wm_name)
    
    count_num=0
    base_num=0
    simi_score_l=[]
    num_queries_l=[]
    budget_l=[]
    result_list=[]
    batch_size=1
    for idx in tqdm(range(0, c4_dataset.data_num, batch_size), ncols=100):#c4_dataset.data_num
        prompt=c4_dataset.data[idx][0]#[0:500]
        wm_text, un_wm_text = wm_scheme.generate(
            prompt, 
            # [tmp[0][0:500] for tmp in c4_dataset.data[idx:idx+batch_size]],
            wm_seed=123, 
            un_wm_flag=True
        )
        wm_text=wm_text[len(prompt):]#[0:500]
        if wm_text[0]==' ':
            wm_text=wm_text[1:]
        un_wm_text=un_wm_text[len(prompt):]#[0:500]
        if un_wm_text[0]==' ':
            un_wm_text=un_wm_text[1:]

        wm_rlt=wm_scheme.detect_wm(wm_text)
        un_rlt=wm_scheme.detect_wm(un_wm_text)
        
        if wm_rlt['is_watermarked']==True and un_rlt['is_watermarked']==False:
            base_num+=1
        else:
            continue
        print()
        print('wm_detect:', wm_rlt)
        print('un_detect:', un_rlt)
        result_list.append({
            'wm_text':wm_text,
            'un_text':un_wm_text,
            'wm_detect':wm_rlt,
            'un_detect':un_rlt,
        })
    
    filename='_'.join([
        wm_name, 
        dataset_name.replace('/','_'), 
        model_name.replace('/','_'), 
        str(c4_dataset.data_num)
    ])+'.json'
    file_path=os.path.join('saved_data', filename)
    save_json(data=result_list, file_path=file_path)

if __name__=="__main__":
    
    file_num=10
    file_data_num=300
    dataset_name='../../dataset/c4/realnewslike'

    text_len = 50
    wm_name = 'SIR'#'SemStamp'
    model_name = "facebook/opt-1.3b"

    for wm_name in ['TS', 'SIR', 'SemStamp', ]:
        get_wm_data(
            file_num=file_num, 
            file_data_num=file_data_num, 
            dataset_name=dataset_name, 
            text_len=text_len, 
            wm_name=wm_name, 
            model_name=model_name
        )
    