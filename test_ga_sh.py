
from textattack.utils import load_json, to_string
import os
from test_ga_attack import test_ga_attack
import argparse
import sys
import numpy as np

def is_debug_mode():
    return sys.gettrace() is not None

def get_key_value(x_dict, key1, key2=None):
    if key2 is not None:
        try:
            value=x_dict[key1][key2]
        except:
            value=x_dict[key1]
    else:
        value=x_dict[key1]
    return value

sh_templte='python test_ga_attack.py --num_generations {num_generations} \
--max_edit_rate {max_edit_rate} --len_weight {len_weight} --eva_thr {eva_thr} \
--fitness_threshold {fitness_threshold} --max_token_num {max_token_num} --victim_tokenizer "{victim_tokenizer}" \
--victim_model "{victim_model}" --wm_name "{wm_name}"  \
--llm_name "{llm_name}" --eva_thr {eva_thr} --mean {mean} \
--std {std} --ab_std {ab_std} --atk_style "{atk_style}" --ori_flag "{ori_flag}" --device {device}  \
--def_stl "{def_stl}" --remove_spoof "{remove_spoof}" --ocr_flag "{ocr_flag}"'

# python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "UPV" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -1 --device 0
# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "UPV" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -1 --device 0
parser = argparse.ArgumentParser(description='test_ga_attack')
parser.add_argument('--llm_name', type=str, default='../model/Llama3.1-8B_hg')
parser.add_argument('--wm_name', type=str, default='KGW')
parser.add_argument('--atk_style', type=str, default='char')
parser.add_argument('--ori_flag', type=str, default='False')
parser.add_argument('--data_aug', type=int, default=9)
parser.add_argument('--ab_std', type=int, default=3)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--max_edit_rate', type=float, default=-1)
parser.add_argument('--remove_spoof', type=str, default='True')
parser.add_argument('--ocr_flag', type=str, default='False')
parser.add_argument('--num_generations', type=int, default=-1)
parser.add_argument('--do_flag', type=str, default='True')
args = parser.parse_args()

do_flag=bool(args.do_flag=='True')
def_stl="spell_check_ltp"
atk_style=args.atk_style
data_aug=args.data_aug
device=args.device
max_edit_rate=args.max_edit_rate
ori_flag=bool(args.ori_flag=='True')
remove_spoof=bool(args.remove_spoof=='True')
ocr_flag=bool(args.ocr_flag=='True')
llm_name=args.llm_name
num_generations=args.num_generations

if 'opt' in llm_name:
    ga_config=load_json(file_path='attk_config/opt_ga_config.json')
else:
    ga_config=load_json(file_path='attk_config/llama_ga_config.json')
max_token_num_list=[100]#,200

if args.wm_name=='':
    wm_name_list=ga_config.keys()
else:
    wm_name_list=[args.wm_name]

if args.ab_std==-1:
    ab_std_list=[1,2,3,4,0,]
else:
    ab_std_list=[args.ab_std]

if ori_flag:
    data_aug=0
    ab_std_list=[100]

for max_token_num in max_token_num_list:
    for wm_name in wm_name_list:
        for ab_std in ab_std_list:
            wm_config=ga_config[wm_name]
            victim_tokenizer=wm_config['victim_tokenizer']
            victim_model=get_key_value(wm_config, 'victim_model', str(data_aug))
            len_weight = get_key_value(wm_config, 'len_weight', str(max_token_num))
            fitness_threshold = get_key_value(wm_config, 'fitness_threshold', str(max_token_num))
            if num_generations<0:
                num_generations=wm_config['num_generations']
            if ocr_flag:
                # num_generations=5
                len_weight=0.0
            eva_thr=wm_config['eva_thr']
            mean=wm_config['mean']
            std=wm_config['std']
            if ab_std==-2:
                ab_std=wm_config['ab_std']
            if max_edit_rate<0:
                max_edit_rate = get_key_value(wm_config, 'max_edit_rate', str(max_token_num))
            
            tmp_sh=sh_templte.format(
                llm_name=llm_name,
                wm_name=wm_name, max_edit_rate=max_edit_rate,  max_token_num=max_token_num, 
                num_generations=num_generations, victim_model=victim_model, 
                victim_tokenizer=victim_tokenizer, 
                len_weight=len_weight,
                fitness_threshold=fitness_threshold, 
                eva_thr=eva_thr, 
                mean=mean, #
                std=std, #
                ab_std=ab_std, #
                atk_style=atk_style, #
                ori_flag=ori_flag, #
                device=device,
                def_stl=def_stl,
                remove_spoof=remove_spoof,
                ocr_flag=ocr_flag
            )
            if do_flag:
                print(tmp_sh)
                if is_debug_mode():
                    print("Running in DEBUG mode")
                    test_ga_attack(
                        llm_name=llm_name, #
                        wm_name=wm_name, 
                        max_edit_rate=max_edit_rate,
                        max_token_num=max_token_num,
                        num_generations=num_generations,
                        victim_model=victim_model,
                        victim_tokenizer=victim_tokenizer,
                        len_weight=len_weight,
                        fitness_threshold=fitness_threshold,
                        eva_thr=eva_thr,
                        mean=mean, #
                        std=std, #
                        ab_std=ab_std, #
                        atk_style=atk_style, #
                        ori_flag=ori_flag, #
                        device=device,
                        def_stl=def_stl,
                        remove_spoof=remove_spoof,
                        ocr_flag=ocr_flag
                    )
                else:
                    print("Running in Normal mode")
                    os.system(tmp_sh)
                
            attk_name='GA'
            if ocr_flag:
                attk_name='GAocr'
            data_records=load_json(
                "saved_attk_data/"+"_".join([
                    attk_name, 
                    str(max_edit_rate), str(num_generations), 
                    str(max_token_num), 
                    str(len_weight),
                    str(fitness_threshold),
                    str(eva_thr),
                    str(mean),
                    str(std),
                    str(float(ab_std)),
                    atk_style,
                    str(ori_flag),
                    def_stl,
                    victim_model.replace('saved_model/',''),
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
            print(to_string([wm_score_drop, asr, token_budget_rate, char_budget_rate, belu, rouge, ppl_rate, adv_ppl], step_char=' '))
            
            if ocr_flag:
                adv_ocr_rate=np.mean([(data_record['ocr_adv_detect']['is_watermarked']==False and data_record['ocr_adv_detect']['is_watermarked']==False) for data_record in data_records])
                print(to_string([adv_ocr_rate], step_char=' '))

# saved_attk_data/GA_0.11_15_100_0.5_0.9_-0.1_1.297289_1.212317757_2.0_token_False_ocr_RefDetector_DIP_.._.._dataset_c4_realnewslike_.._model_Llama3.1-8B_hg_bert-base-uncased_2025-02-13.json
# saved_attk_data/GA_0.11_15_100_0.5_0.9_-0.1_1.297289_1.212317757_2  _token_False_ocr_RefDetector_DIP_.._.._dataset_c4_realnewslike_.._model_Llama3.1-8B_hg_bert-base-uncased_2025-02-13.json