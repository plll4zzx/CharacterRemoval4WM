
from textattack.utils import load_json, to_string, compute_auc, compute_log_auc
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
parser.add_argument('--wm_name_list', type=str, default="['KGW','DIP', 'SynthID','Unigram','Unbiased']")
parser.add_argument('--atk_style_list', type=str, default="['token','char','sentence']")
parser.add_argument('--ori_flag', type=str, default='False')
parser.add_argument('--data_aug_list', type=str, default="[9]")
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--max_edit_rate_list', type=str, default="[0.05, 0.1,0.15,0.2,0.3,0.4,0.5]")
parser.add_argument('--def_stl', type=str, default='')#"ocr"#spell_check_ltp
parser.add_argument('--do_flag', type=str, default='False')
parser.add_argument('--auc_flag', type=str, default='False')
parser.add_argument('--atk_times_list', type=str, default='[1,10,50,100]')
parser.add_argument('--max_token_num_list', type=str, default='[100]')
args = parser.parse_args()

# do_flag=True
# do_flag=False
# ori_flag="False"#"True"#
# auc_flag="True"#"False"#
do_flag=bool(args.do_flag=='True')
llm_name=args.llm_name#'../model/Llama3.1-8B_hg'#'facebook/opt-1.3b'#
ori_flag=bool(args.ori_flag=='True')
auc_flag=bool(args.auc_flag=='True')

def_stl=args.def_stl

atk_style_list=eval(args.atk_style_list)
atk_times_list=eval(args.atk_times_list)
max_token_num_list=eval(args.max_token_num_list)
device=0

data_aug_list=eval(args.data_aug_list)
max_edit_rate_list=eval(args.max_edit_rate_list)
wm_name_list=eval(args.wm_name_list)

if 'opt' in llm_name:
    rand_config=load_json(file_path='attk_config/opt_rand_config.json')
else:
    rand_config=load_json(file_path='attk_config/llama_rand_config.json')
char_op=2

for max_token_num in max_token_num_list:
    for atk_style in atk_style_list:
        for wm_name in wm_name_list:
            wm_config=rand_config[wm_name]
            ref_tokenizer=wm_config['ref_tokenizer']
            
            print(to_string([llm_name, wm_name, atk_style, char_op], step_char='\t'))
            for data_aug in data_aug_list:
                if data_aug==-1:
                    tmp_ori_flag="True"
                    tmp_data_aug=9
                else:
                    tmp_ori_flag=ori_flag
                    tmp_data_aug=data_aug
                ref_model=wm_config['ref_model'][str(tmp_data_aug)]
                for max_edit_rate in max_edit_rate_list:
                    for atk_times in atk_times_list:
                        if 'sand' in atk_style:
                            atk_times=1
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
                                ref_model.replace('saved_model/','')
                            ])+".json"
                        )

                        un_records=load_json(
                            "saved_attk_data/"+"_".join([
                                'un_text',
                                llm_name.replace('/','_'), wm_name.replace('/','_'), 
                                str(max_token_num), 
                            ])+".json"
                        )
                        
                        un_score=[data_record['un_detect']['score'] for data_record in un_records]
                        wm_score=[data_record['wm_detect']['score'] for data_record in data_records]
                        adv_score=[data_record['adv_detect']['score'] for data_record in data_records]
                        wm_score_min=np.min(un_score)
                        wm_score_max=np.max(wm_score)
                        pre_auc=compute_auc(wm_score, un_score)
                        after_auc=compute_auc(adv_score, un_score)
                        if auc_flag:
                            print(np.min(adv_score), np.max(adv_score), pre_auc, after_auc)
                            continue
                        wm_score_drop=np.mean([data_record['wm_score_drop']/(wm_score_max-wm_score_min) for data_record in data_records])
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
                        print(to_string(['wm_score_drop', 'asr', 'token_budget_rate', 'char_budget_rate', 'belu', 'rouge', 'ppl_rate', 'adv_ppl', 'ref_drop'], step_char=' '))
                        print(to_string([wm_score_drop, asr, token_budget_rate, char_budget_rate, belu, rouge, ppl_rate, adv_ppl, ref_drop], step_char=' '))
                        if len(def_stl)>0:
                            adv_ocr_rate=np.mean([
                                (
                                    data_record['ocr_adv_detect']['is_watermarked']==False 
                                    # and data_record['adv_detect']['is_watermarked']==False
                                ) 
                                for data_record in data_records
                            ])
                            print(to_string([adv_ocr_rate], step_char=' '))
