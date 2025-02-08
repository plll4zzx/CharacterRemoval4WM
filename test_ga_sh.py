
from textattack.utils import load_json
import os
from test_ga_attack import test_ga_attack
import argparse

def get_key_value(x_dict, key1, key2=None):
    if key2 is not None:
        try:
            value=x_dict[key1][key2]
        except:
            value=x_dict[key1]
    else:
        value=x_dict[key1]
    return value

sh_templte='python test_ga_attack.py --num_generations {num_generations} --max_edit_rate {max_edit_rate} --len_weight {len_weight} --eva_thr {eva_thr} --fitness_threshold {fitness_threshold} --max_token_num {max_token_num} --victim_tokenizer "{victim_tokenizer}" --victim_model "{victim_model}" --wm_name "{wm_name}"'

# python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "UPV"
# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "UPV"
parser = argparse.ArgumentParser(description='test_ga_attack')
parser.add_argument('--llm_name', type=str, default='facebook/opt-1.3b')
parser.add_argument('--wm_name', type=str, default='SynthID')
args = parser.parse_args()

llm_name=args.llm_name
if 'opt' in llm_name:
    ga_config=load_json(file_path='attk_config/opt_ga_config.json')
else:
    ga_config=load_json(file_path='attk_config/llama_ga_config.json')
max_token_num_list=[100,200]#

for max_token_num in max_token_num_list:
    for wm_name in [args.wm_name]:#rand_config:
        wm_config=ga_config[wm_name]
        victim_tokenizer=wm_config['victim_tokenizer']
        victim_model=wm_config['victim_model']
        num_generations=wm_config['num_generations']
        eva_thr=wm_config['eva_thr']
        mean=wm_config['mean']
        std=wm_config['std']
        ab_std=wm_config['ab_std']
        max_edit_rate = get_key_value(wm_config, 'max_edit_rate', str(max_token_num))
        len_weight = get_key_value(wm_config, 'len_weight', str(max_token_num))
        fitness_threshold = get_key_value(wm_config, 'fitness_threshold', str(max_token_num))
        
        tmp_sh=sh_templte.format(
            num_generations=num_generations, len_weight=len_weight, eva_thr=eva_thr, 
            fitness_threshold=fitness_threshold, max_edit_rate=max_edit_rate, max_token_num=max_token_num, 
            victim_tokenizer=victim_tokenizer, victim_model=victim_model, wm_name=wm_name
        )
        print(tmp_sh)
        # os.system(tmp_sh)
        test_ga_attack(
            llm_name=llm_name,
            wm_name=wm_name, 
            max_edit_rate=max_edit_rate,
            max_token_num=max_token_num,
            num_generations=num_generations,
            victim_model=victim_model,
            victim_tokenizer=victim_tokenizer,
            len_weight=len_weight,
            fitness_threshold=fitness_threshold,
            eva_thr=eva_thr,
            mean=mean,
            std=std,
            ab_std=ab_std
        )

