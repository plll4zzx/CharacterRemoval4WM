
import torch
import json
from MarkLLM.watermark.auto_watermark import AutoWatermark
from MarkLLM.utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

import textattack
from read_data import c4
import textattack.attack_sems
import numpy as np

class LLM_WM:

    def __init__(self, model_name = "facebook/opt-1.3b", device = "cuda", wm_name='KGW'):
        self.model_name = model_name
        self.wm_name=wm_name
        self.device = device if torch.cuda.is_available() else "cpu"

        # Transformers config
        self.transformers_config = TransformersConfig(
            model=AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device),
            tokenizer=AutoTokenizer.from_pretrained(self.model_name),
            vocab_size=50272,
            device=self.device,
            max_new_tokens=200,
            min_length=230,
            do_sample=True,
            no_repeat_ngram_size=4
        )

        # Load watermark algorithm
        self.wm_model = AutoWatermark.load(f'{self.wm_name}', algorithm_config=f'MarkLLM/config/{self.wm_name}.json', transformers_config=self.transformers_config)
    
    def generate(self, prompt):
        # Generate text
        wm_text = self.wm_model.generate_watermarked_text(prompt)
        un_wm_text = self.wm_model.generate_unwatermarked_text(prompt)
        return wm_text, un_wm_text

    def detect_wm(self, text):
        # Detect
        result = self.wm_model.detect_watermark(text)
        return result

if __name__=="__main__":
    
    file_num=10
    file_data_num=2
    dataset_name='../../dataset/c4/realnewslike'
    file_num=int(file_num)


    text_len=50
    c4_dataset=c4(dir_path=dataset_name, file_num=file_num, file_data_num=file_data_num)
    c4_dataset.load_data(text_len)

    wm_scheme=LLM_WM(model_name = "facebook/opt-1.3b", device = "cuda", wm_name='SIR')
    
    target_cos=0.5
    edit_distance=10
    attack_name = 'TextBuggerLi2018'
    victim_name = 'sentence-transformers/all-mpnet-base-v2'
    model = transformers.AutoModel.from_pretrained(victim_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(victim_name)
    model_wrapper = textattack.models.wrappers.HuggingFaceEncoderWrapper(model, tokenizer)
    attack = getattr(textattack.attack_sems, attack_name).build(model_wrapper, target_cos=target_cos, edit_distance=edit_distance)
    # attack_args = textattack.AttackArgs(
    #     num_examples=int(file_num*file_data_num),
    #     attack_name=attack_name,
    #     dataset_name=dataset_name.replace('/','_'),
    #     victim_name=victim_name,
    #     query_budget=1000,
    #     disable_stdout=False,
    #     parallel=False
    # )
    # attacker = textattack.Attacker(attack, [], attack_args)
    count_num=0
    base_num=0
    for idx in range(10):
        wm_text, un_wm_text = wm_scheme.generate(c4_dataset.data[1+idx][0][0:500])
        wm_text=wm_text[0:1000]
        un_wm_text=un_wm_text[0:1000]
        print('wm_text')
        print(wm_text)
        wm_rlt=wm_scheme.detect_wm(wm_text)
        print(wm_rlt)
        un_wm_rlt=wm_scheme.detect_wm(un_wm_text)
        print(un_wm_rlt)
        attacked=attack.step_attack(wm_text, 0, window_size=10, step_ize=10)
        print(round(np.mean(attacked['score']),4))
        print('attacked_text')
        print(attacked['text'])
        attacked_rlt=wm_scheme.detect_wm(attacked['text'])
        print(attacked_rlt)

        if wm_rlt['is_watermarked']==True and un_wm_rlt['is_watermarked']==False:
            base_num+=1
            if attacked_rlt['is_watermarked']==False:
                count_num+=1
    print(count_num, base_num)