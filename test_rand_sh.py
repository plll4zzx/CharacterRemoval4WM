
from textattack.utils import load_json
import os
from test_random_attack import test_rand_attack
import argparse

sh_templte='python test_random_attack.py --atk_style "{atk_style}" --max_edit_rate {max_edit_rate} --atk_times {atk_times} \
--max_token_num {max_token_num} --ref_tokenizer "{ref_tokenizer}" --ref_model "{ref_model}" --wm_name "{wm_name}" \
--llm_name "{llm_name}" --ori_flag "{ori_flag}" --def_stl "{def_stl}" --device {device}'

parser = argparse.ArgumentParser(description='test_rand_attack')
parser.add_argument('--llm_name', type=str, default='facebook/opt-1.3b')
parser.add_argument('--wm_name', type=str, default='')
parser.add_argument('--atk_style', type=str, default='char')
parser.add_argument('--ori_flag', type=str, default='False')
parser.add_argument('--data_aug', type=int, default=9)
parser.add_argument('--ab_std', type=int, default=1)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--max_edit_rate', type=float, default=-1)
args = parser.parse_args()


llm_name='facebook/opt-1.3b'#'../model/Llama3.1-8B_hg'#
data_aug=9
ori_flag="False"
def_stl=""
atk_style_list=['char']#['low','ende', 'mix_char']#,'char','token', 'BERTAttackLi2020',
atk_times_list=[10]#1,,50,100
max_token_num_list=[100]#100,50, 100,150,  
device=1

if 'opt' in llm_name:
    rand_config=load_json(file_path='attk_config/opt_rand_config.json')
else:
    rand_config=load_json(file_path='attk_config/llama_rand_config.json')

for data_aug in [9]:
    for max_token_num in max_token_num_list:
        for wm_name in ['DIP']:#rand_config:#,'Unbiased''DIP', 'SynthID','Unigram','Unbiased','KGW'
            wm_config=rand_config[wm_name]
            ref_tokenizer=wm_config['ref_tokenizer']
            if data_aug==-1:
                ori_flag="True"
                data_aug=9
            ori_flag=bool(ori_flag=='True')
            ref_model=wm_config['ref_model'][str(data_aug)]
            max_edit_rate_list=[0.1]#wm_config['max_edit_rate']0.05,0.1,
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
                            ori_flag=ori_flag,
                            def_stl=def_stl,
                            device=device
                        )
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
                            atk_times=atk_times,
                            ori_flag=ori_flag,
                            def_stl=def_stl,
                            device=device
                        )
