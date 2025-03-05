
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

sh_templte='python test_ga_attack.py --num_generations {num_generations} \
--max_edit_rate {max_edit_rate} --len_weight {len_weight} --eva_thr {eva_thr} \
--fitness_threshold {fitness_threshold} --max_token_num {max_token_num} --victim_tokenizer "{victim_tokenizer}" \
--victim_model "{victim_model}" --wm_name "{wm_name}"  \
--llm_name "{llm_name}" --eva_thr {eva_thr} --mean {mean} \
--std {std} --ab_std {ab_std} --atk_style "{atk_style}" --ori_flag "{ori_flag}" --device {device}  --def_stl "{def_stl}"'

# python test_ga_sh.py --llm_name "facebook/opt-1.3b" --wm_name "UPV" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -1 --device 0
# python test_ga_sh.py --llm_name "../model/Llama3.1-8B_hg" --wm_name "UPV" --atk_style "char" --ori_flag "False" --data_aug 9 --ab_std -1 --device 0
parser = argparse.ArgumentParser(description='test_ga_attack')
parser.add_argument('--llm_name', type=str, default='facebook/opt-1.3b')
parser.add_argument('--wm_name', type=str, default='')
parser.add_argument('--atk_style', type=str, default='char')
parser.add_argument('--ori_flag', type=str, default='False')
parser.add_argument('--data_aug', type=int, default=9)
parser.add_argument('--ab_std', type=int, default=1)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--max_edit_rate', type=float, default=-1)
args = parser.parse_args()

def_stl="ocr"
atk_style=args.atk_style
data_aug=args.data_aug
device=args.device
max_edit_rate=args.max_edit_rate
ori_flag=bool(args.ori_flag=='True')
llm_name=args.llm_name
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
    ab_std_list=[0,1,2,3,4]
else:
    ab_std_list=[args.ab_std]

for max_token_num in max_token_num_list:
    for wm_name in wm_name_list:
        for ab_std in ab_std_list:
            wm_config=ga_config[wm_name]
            victim_tokenizer=wm_config['victim_tokenizer']
            victim_model=get_key_value(wm_config, 'victim_model', str(data_aug))
            num_generations=wm_config['num_generations']
            eva_thr=wm_config['eva_thr']
            mean=wm_config['mean']
            std=wm_config['std']
            # ab_std=wm_config['ab_std']
            if max_edit_rate<0:
                max_edit_rate = get_key_value(wm_config, 'max_edit_rate', str(max_token_num))
            len_weight = get_key_value(wm_config, 'len_weight', str(max_token_num))
            fitness_threshold = get_key_value(wm_config, 'fitness_threshold', str(max_token_num))
            
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
            )
            print(tmp_sh)
            # os.system(tmp_sh)
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
            )

