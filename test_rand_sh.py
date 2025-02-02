
from textattack.utils import load_json
import os
from test_random_attack import test_rand_attack

sh_templte='python test_random_attack.py --atk_style "{atk_style}" --max_edit_rate {max_edit_rate} --atk_times {atk_times} --max_token_num {max_token_num} --ref_tokenizer "{ref_tokenizer}" --ref_model "{ref_model}" --wm_name "{wm_name}"'

llm_name='../model/Llama3.1-8B_hg'#'facebook/opt-1.3b'
rand_config=load_json(file_path='attk_config/llama_rand_config.json')
atk_style_list=['token', 'char', 'mix_char']#['low','ende']#
atk_times_list=[1,10,50,100]
max_token_num_list=[100,200]#

for max_token_num in max_token_num_list:
    for wm_name in ['Unigram']:#rand_config:#,'Unbiased'
        wm_config=rand_config[wm_name]
        ref_tokenizer=wm_config['ref_tokenizer']
        ref_model=wm_config['ref_model']
        max_edit_rate_list=[0.05,0.1]#wm_config['max_edit_rate']
        for atk_style in atk_style_list:
            for max_edit_rate in max_edit_rate_list:
                for atk_times in atk_times_list:
                    tmp_sh=sh_templte.format(atk_style=atk_style, max_edit_rate=max_edit_rate, atk_times=atk_times, max_token_num=max_token_num, ref_tokenizer=ref_tokenizer, ref_model=ref_model, wm_name=wm_name)
                    print(tmp_sh)
                    # os.system(tmp_sh)
                    test_rand_attack(
                        llm_name=llm_name,
                        wm_name=wm_name, 
                        max_edit_rate=max_edit_rate,
                        max_token_num=max_token_num,
                        atk_style=atk_style,
                        ref_tokenizer=ref_tokenizer,
                        ref_model=ref_model,
                        atk_times=atk_times
                    )
