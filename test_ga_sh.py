
from textattack.utils import load_json
import os

sh_templte='python test_ga_attack.py --num_generations {num_generations} --max_edit_rate {max_edit_rate} --len_weight {len_weight} --eva_thr {eva_thr} --fitness_threshold {fitness_threshold} --max_token_num {max_token_num} --victim_tokenizer "{victim_tokenizer}" --victim_model "{victim_model}" --wm_name "{wm_name}"'

ga_config=load_json(file_path='ga_config.json')
max_token_num_list=[100]#,200

for max_token_num in max_token_num_list:
    for wm_name in ['SynthID']:#rand_config:
        wm_config=ga_config[wm_name]
        victim_tokenizer=wm_config['victim_tokenizer']
        victim_model=wm_config['victim_model']
        max_edit_rate=wm_config['max_edit_rate']
        num_generations=wm_config['num_generations']
        len_weight=wm_config['len_weight']
        fitness_threshold=wm_config['fitness_threshold']
        eva_thr=wm_config['eva_thr']
        tmp_sh=sh_templte.format(num_generations=num_generations, len_weight=len_weight, eva_thr=eva_thr, fitness_threshold=fitness_threshold, max_edit_rate=max_edit_rate, max_token_num=max_token_num, victim_tokenizer=victim_tokenizer, victim_model=victim_model, wm_name=wm_name)
        print(tmp_sh)
        os.system(tmp_sh)
