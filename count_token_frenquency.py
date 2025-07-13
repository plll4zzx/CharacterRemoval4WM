
from textattack.utils import to_string, load_json, save_json
from llm_wm import LLM_WM
import os

def count_fq(
    wm_name, 
    llm_name="facebook/opt-1.3b",
    dataset_name='../../dataset/c4/realnewslike',
):
    wm_data=load_json("saved_data/"+"_".join([wm_name, dataset_name.replace('/','_'), llm_name.replace('/','_')])+"_5000.json")
    device="cuda:0"
    wm_scheme=LLM_WM(model_name = llm_name, device = device, wm_name=wm_name)

    wm_token_rate={}
    un_token_rate={}
    num_w,num_u=0,0
    for idx in range(len(wm_data)):
        wm_text=wm_data[idx]['wm_text']
        wm_tokens=wm_scheme.transformers_config.tokenizer.tokenize(wm_text)
        wm_tokens=[token.lstrip("Ġ▁") for token in wm_tokens]
        for tmp_token in wm_tokens:
            if tmp_token in wm_token_rate:
                wm_token_rate[tmp_token]+=1
            else:
                wm_token_rate[tmp_token]=1
            num_w+=1
        un_text=wm_data[idx]['un_text']
        un_tokens=wm_scheme.transformers_config.tokenizer.tokenize(un_text)
        un_tokens=[token.lstrip("Ġ▁") for token in un_tokens]
        for tmp_token in un_tokens:
            if tmp_token in un_token_rate:
                un_token_rate[tmp_token]+=1
            else:
                un_token_rate[tmp_token]=1
            num_u+=1
    token_rate={}
    for tmp_token in wm_token_rate:
        if tmp_token in un_token_rate:
            token_rate[tmp_token]=(wm_token_rate[tmp_token]/num_w)/(un_token_rate[tmp_token]/num_u)
        else:
            token_rate[tmp_token]=(wm_token_rate[tmp_token]/num_w)/(0.5/num_u)
    
    filename='_'.join([
        'TokenFq',wm_name, 
        dataset_name.replace('/','_'), 
        llm_name.replace('/','_'), 
    ])+'.json'
    file_path=os.path.join('saved_data', filename)
    save_json(data=token_rate, file_path=file_path)
    
if __name__=='__main__':
    wm_name="KGW"
    llm_name="facebook/opt-1.3b"
    dataset_name='../../dataset/c4/realnewslike'
    for wm_name in ['KGW','DIP', 'SynthID','Unigram','Unbiased']:
        for llm_name in ['facebook/opt-1.3b','../model/Llama3.1-8B_hg']:
            count_fq(wm_name=wm_name, llm_name=llm_name, dataset_name=dataset_name)