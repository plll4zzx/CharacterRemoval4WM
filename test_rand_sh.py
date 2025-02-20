
from textattack.utils import load_json
import os
from test_random_attack import test_rand_attack

sh_templte='python test_random_attack.py --atk_style "{atk_style}" --max_edit_rate {max_edit_rate} --atk_times {atk_times} --max_token_num {max_token_num} --ref_tokenizer "{ref_tokenizer}" --ref_model "{ref_model}" --wm_name "{wm_name}" --llm_name "{llm_name}" --ori_flag "{ori_flag}"'

llm_name='../model/Llama3.1-8B_hg'#'facebook/opt-1.3b'#

if 'opt' in llm_name:
    rand_config=load_json(file_path='attk_config/opt_rand_config.json')
else:
    rand_config=load_json(file_path='attk_config/llama_rand_config.json')
data_aug=9
ori_flag="False"
atk_style_list=['BERTAttackLi2020']#['low','ende', 'mix_char']#,'char','token', ,'TextBuggerLi2018'
atk_times_list=[1]#1,,50,100
max_token_num_list=[100]#100,50, 100,150,  

for data_aug in [9]:
    for max_token_num in max_token_num_list:
        for wm_name in ['KGW']:#rand_config:#,'Unbiased'['DIP', 'SynthID','Unigram','Unbiased']:#
            wm_config=rand_config[wm_name]
            ref_tokenizer=wm_config['ref_tokenizer']
            if data_aug==-1:
                ori_flag="True"
                data_aug=9
            ref_model=wm_config['ref_model'][str(data_aug)]
            max_edit_rate_list=[0.2]#wm_config['max_edit_rate']0.05,0.1,
            for atk_style in atk_style_list:
                for max_edit_rate in max_edit_rate_list:
                    for atk_times in atk_times_list:
                        tmp_sh=sh_templte.format(
                            atk_style=atk_style, 
                            max_edit_rate=max_edit_rate, 
                            atk_times=atk_times, 
                            max_token_num=max_token_num, 
                            ref_tokenizer=ref_tokenizer, 
                            ref_model=ref_model, 
                            wm_name=wm_name,
                            llm_name=llm_name,
                            ori_flag=ori_flag
                        )
                        print(tmp_sh)
                        os.system(tmp_sh)
                        # test_rand_attack(
                        #     llm_name=llm_name,
                        #     wm_name=wm_name, 
                        #     max_edit_rate=max_edit_rate,
                        #     max_token_num=max_token_num,
                        #     atk_style=atk_style,
                        #     ref_tokenizer=ref_tokenizer,
                        #     ref_model=ref_model,
                        #     atk_times=atk_times,
                        #     ori_flag=ori_flag,
                        # )
