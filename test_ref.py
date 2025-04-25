from textattack.utils import load_json, to_string
import os
from test_random_attack import test_rand_attack
import argparse
import numpy as np
import sys

sh_templte='python train_ref_detector.py  --wm_name "{wm_name}" --ths {ths}  --llm_name "{llm_name}" --num_epochs {num_epochs} --rand_char_rate {rand_char_rate} --model_path "{model_path}" --device {device} --rand_times {rand_times}'


llm_name='../model/Llama3.1-8B_hg'#'facebook/opt-1.3b'#
if 'opt' in llm_name:
    rand_config=load_json(file_path='attk_config/opt_rand_config.json')
else:
    rand_config=load_json(file_path='attk_config/llama_rand_config.json')
num_epochs=0
device=0
rand_char_rate=0.15

ths_dict={
    'KGW':4,
    'DIP':1.513,
    'SynthID':0.52,
    'Unigram':4,
    'Unbiased':1.513,
}

for data_aug in [0,5,9]:
    for wm_name in ['KGW','DIP']:#,rand_config:#,'KGW','DIP', 'SynthID','Unigram','Unbiased'
        wm_config=rand_config[wm_name]
        model_path=wm_config['ref_model'][str(data_aug)]
        ths=ths_dict[wm_name]
        tmp_sh=sh_templte.format( 
            wm_name=wm_name,
            llm_name=llm_name,
            num_epochs=num_epochs,
            model_path=model_path,
            rand_char_rate=rand_char_rate,
            device=device,
            rand_times=data_aug,
            ths=ths
        )
        print(tmp_sh)
        os.system(tmp_sh)