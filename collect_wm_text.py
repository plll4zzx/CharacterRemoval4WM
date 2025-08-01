
from read_data import c4
from textattack.utils import to_string, save_json, get_advtext_filename
from llm_wm import LLM_WM
import os
from tqdm import tqdm
import argparse
import torch
# from semantic_attack import SemanticAttack
import numpy as np
# from deep_translator import GoogleTranslator
from transformers import pipeline
# import datetime
# import textattack.attack_sems

def get_wm_data(
    file_num=10,
    file_data_num=20,
    dataset_name='../../dataset/c4/realnewslike',
    text_len=50,
    wm_name='SIR',#'SemStamp'
    model_name = "facebook/opt-1.3b",
    rand_seed=123,
    device=0,
    context_len=1,
    language='en',
):
    print(language)
    device = "cuda:"+str(device) if torch.cuda.is_available() else "cpu"
    if language!='en':
        # translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
        # translator = GoogleTranslator(source='auto', target=language)
        translator = pipeline(
            "translation",
            model="facebook/mbart-large-50-many-to-many-mmt",
            tokenizer="facebook/mbart-large-50-many-to-many-mmt",
            src_lang="en_XX",
            tgt_lang="fr_XX",
            device=device 
        )

    c4_dataset=c4(dir_path=dataset_name, file_num=file_num, file_data_num=file_data_num, rand_seed=rand_seed)
    c4_dataset.load_data(text_len)
    wm_scheme=LLM_WM(model_name = model_name, device = device, wm_name=wm_name, context_len=context_len)
    
    count_num=0
    base_num=0
    simi_score_l=[]
    num_queries_l=[]
    budget_l=[]
    result_list=[]
    batch_size=1
    len_list=[]
    for idx in tqdm(range(0, c4_dataset.data_num, batch_size), ncols=100):#c4_dataset.data_num
        prompt=c4_dataset.data[idx][0]#[0:500]
        if language!='en':
            translated = translator(prompt)[0]['translation_text']
            prompt=translated
        token_id_list=wm_scheme.transformers_config.tokenizer.encode(prompt, add_special_tokens=False)
        prompt=wm_scheme.transformers_config.tokenizer.decode(token_id_list[0:50], skip_special_tokens=True)
        wm_text, un_wm_text = wm_scheme.generate(
            prompt, 
            wm_seed=123, 
            un_wm_seed=123, 
            un_wm_flag=True
        )

        if wm_text[0:len(prompt)]==prompt:
            wm_text=wm_text[len(prompt):]
        if un_wm_text[0:len(prompt)]==prompt:
            un_wm_text=un_wm_text[len(prompt):]
        
        if wm_text[0]==' ':
            wm_text=wm_text[1:]
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
    if language != 'en':
        wm_name=wm_name+language
    filename='_'.join([
        wm_name, 
        dataset_name.replace('/','_'), 
        model_name.replace('/','_'), 
        str(c4_dataset.data_num), 
        # str(3)
    ])+'.json'
    file_path=os.path.join('saved_data', filename)
    save_json(data=result_list, file_path=file_path)

if __name__=="__main__":
    
    file_num=10
    file_data_num=500
    dataset_name='../../dataset/c4/realnewslike'
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"#"../model/Llama3.1-8B_hg"#"facebook/opt-1.3b"meta-llama/Llama-3.1-8B-Instruct

    text_len = 50
    # wm_name = 'SIR'#'SemStamp'
    rand_seed=123

    parser = argparse.ArgumentParser(description='collect wm data')
    parser.add_argument('--dataset_name', type=str, default='../../dataset/c4/realnewslike')
    parser.add_argument('--model_name', type=str, default="../model/Llama3.1-8B_hg")
    parser.add_argument('--wm_name', type=str, default="KGW")
    parser.add_argument('--file_num', type=int, default=10)
    parser.add_argument('--file_data_num', type=int, default=100)
    parser.add_argument('--text_len', type=int, default=50)
    parser.add_argument('--rand_seed', type=int, default=123)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--context_len', type=int, default=1)
    parser.add_argument('--language', type=str, default="fr")

    args = parser.parse_args()
    # for wm_name in ['Unbiased', 'KGW', 'Unigram']:#'SIR', 'SemStamp', 
    get_wm_data(
        file_num=args.file_num, 
        file_data_num=args.file_data_num, 
        dataset_name=args.dataset_name, 
        text_len=args.text_len, 
        wm_name=args.wm_name, 
        model_name=args.model_name,
        rand_seed=args.rand_seed,
        device=args.device,
        context_len=args.context_len,
        language=args.language
    )
    