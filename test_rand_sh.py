
from textattack.utils import load_json, to_string
import os
from test_random_attack import test_rand_attack
import argparse
import numpy as np
import sys

def is_debug_mode():
    return sys.gettrace() is not None

sh_templte='python test_random_attack.py --atk_style "{atk_style}" --max_edit_rate {max_edit_rate} --atk_times {atk_times} \
--max_token_num {max_token_num} --ref_tokenizer "{ref_tokenizer}" --ref_model "{ref_model}" --wm_name "{wm_name}" \
--llm_name "{llm_name}" --ori_flag "{ori_flag}" --def_stl "{def_stl}" --device {device} --char_op {char_op}'

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

do_flag=True
# do_flag=False
llm_name='facebook/opt-1.3b'#'../model/Llama3.1-8B_hg'#
data_aug=9
ori_flag="False"
def_stl="spell_check_ltp"
atk_style_list=['char',]#['low','ende', 'mix_char']#,'char','token', 'BERTAttackLi2020',
atk_times_list=[1]#1,,50,100
max_token_num_list=[100]#100, 50, 100, 150, 200 
device=1

if 'opt' in llm_name:
    rand_config=load_json(file_path='attk_config/opt_rand_config.json')
else:
    rand_config=load_json(file_path='attk_config/llama_rand_config.json')
char_op=6
for max_token_num in max_token_num_list:
    for atk_style in atk_style_list:
        for wm_name in ['KGW','DIP', 'SynthID','Unigram','Unbiased']:#,'SynthID',rand_config:#,'KGW','DIP', 'SynthID','Unigram','Unbiased'
            wm_config=rand_config[wm_name]
            ref_tokenizer=wm_config['ref_tokenizer']
            ori_flag=bool(ori_flag=='True')
            max_edit_rate_list=[0.2]#,0.2wm_config['max_edit_rate'][0.05,0.1,0.15,0.2,0.3,0.4,0.5
            print(to_string([wm_name, atk_style, char_op], step_char='\t'))
            for data_aug in [0,5,9,-1]:
                if data_aug==-1:
                    tmp_ori_flag="True"
                    data_aug=9
                else:
                    tmp_ori_flag=ori_flag
                ref_model=wm_config['ref_model'][str(data_aug)]
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
                            ori_flag=tmp_ori_flag,
                            def_stl=def_stl,
                            device=device,
                            char_op=char_op,
                        )
                        if do_flag:
                            print(tmp_sh)
                            if is_debug_mode():
                                print("Running in DEBUG mode")
                                test_rand_attack(
                                    llm_name=llm_name,
                                    wm_name=wm_name, 
                                    max_edit_rate=max_edit_rate,
                                    max_token_num=max_token_num,
                                    atk_style=atk_style,
                                    ref_tokenizer=ref_tokenizer,
                                    ref_model=ref_model,
                                    atk_times=atk_times,
                                    ori_flag=tmp_ori_flag,
                                    def_stl=def_stl,
                                    device=device,
                                    char_op=char_op,
                                )
                            else:
                                print("Running in Normal mode")
                                os.system(tmp_sh)
                        
                        attk_name='Rand'
                        if char_op!=2 and 'char' in atk_style:
                            attk_name='RandChar_'+str(char_op)
                            tmp_atk_style='char'
                        else:
                            tmp_atk_style=atk_style
                        data_records=load_json(
                            "saved_attk_data/"+"_".join([
                                attk_name, 
                                # llm_name.replace('/','_'), wm_name.replace('/','_'), 
                                str(max_edit_rate), str(max_token_num), tmp_atk_style, 
                                str(atk_times), str(tmp_ori_flag), def_stl, 
                                ref_model.replace('saved_model/',''), 
                            ])+".json"
                        )
                        
                        wm_score_drop=np.mean([data_record['wm_score_drop']/data_record['wm_detect']['score'] for data_record in data_records])
                        asr=np.mean([data_record['adv_detect']['is_watermarked']==False for data_record in data_records])
                        token_budget_rate=np.mean([data_record['t_edit_dist']/data_record['token_num'] for data_record in data_records])
                        char_budget_rate=np.mean([data_record['c_edit_dist']/data_record['char_num'] for data_record in data_records])
                        belu=np.mean([data_record['belu'] for data_record in data_records])
                        rouge=np.mean([data_record['rouge-f1'] for data_record in data_records])
                        ppl_rate=np.mean([data_record['ppl_rate'] for data_record in data_records])
                        adv_ppl=np.mean([data_record['adv_ppl'] for data_record in data_records])
                        wm_ref_score_l=[data_record['wm_ref_score'] for data_record in data_records]
                        wm_ref_score=np.mean(wm_ref_score_l)-np.min(wm_ref_score_l)

                        adv_ref_score_l=[data_record['adv_ref_score'] for data_record in data_records]
                        adv_ref_score=np.mean(adv_ref_score_l)-np.min(adv_ref_score_l)
                        ref_drop=(wm_ref_score-adv_ref_score)/wm_ref_score
                        print(to_string([wm_score_drop, asr, token_budget_rate, char_budget_rate, belu, rouge, ppl_rate, adv_ppl, ref_drop], step_char=' '))
