
from textattack.utils import load_json
import os

sh_templte='python test_random_attack.py --atk_style "{atk_style}" --max_edit_rate {max_edit_rate} --atk_times {atk_times} --max_token_num {max_token_num} --ref_tokenizer "{ref_tokenizer}" --ref_model "{ref_model}" --wm_name "{wm_name}"'

rand_config=load_json(file_path='rand_config.json')
atk_style_list=['token', 'char', 'mix_char']
atk_times_list=[1,10,50,100]
max_token_num_list=[200]#100,

for max_token_num in max_token_num_list:
    for wm_name in ['Unigram','Unbiased']:#rand_config:
        wm_config=rand_config[wm_name]
        ref_tokenizer=wm_config['ref_tokenizer']
        ref_model=wm_config['ref_model']
        max_edit_rate_list=wm_config['max_edit_rate']
        for atk_style in atk_style_list:
            for max_edit_rate in max_edit_rate_list:
                for atk_times in atk_times_list:
                    tmp_sh=sh_templte.format(atk_style=atk_style, max_edit_rate=max_edit_rate, atk_times=atk_times, max_token_num=max_token_num, ref_tokenizer=ref_tokenizer, ref_model=ref_model, wm_name=wm_name)
                    print(tmp_sh)
                    os.system(tmp_sh)
