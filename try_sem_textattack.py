import importlib
import json
import os
import time

import textattack
import transformers

import textattack.attack_sems
# from datasets import load_dataset
# from textattack.models.helpers import (LSTMForClassification,
                                    #    WordCNNForClassification)

# from utils.func import use_proxy

# pid = os.getpid()
# print(pid)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# #advdata/imdb/ori_testTextFoolerbert_imdb是用原始Hugging faces imdb
# #BAE在原始的huggingface imdb上生成不了对抗样本，改为在原始的huggingface imdb tokenize<512之后的ori_hug_token_imdb文件上生成对抗样本

def main(
    dataset_name = 'mnli',
    attack_name = 'TextFoolerJin2019', 
    victim_name = 'sentence-transformers/all-mpnet-base-v2',
):#['BAEGarg2019', 'TextBuggerLi2018', 'PWWSRen2019', 'DeepWordBugGao2018']

    data=[("I enjoyed the movie a lot!", 1), ("Absolutely horrible film.", 0), ("Our family had a fun time!", 1)]

    tokenizer =transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
    
    model_wrapper = textattack.models.wrappers.HuggingFaceEncoderWrapper(model, tokenizer)
    dataset = textattack.datasets.Dataset(data)
    
    attack = getattr(textattack.attack_sems, attack_name).build(model_wrapper)
    model = transformers.AutoModel.from_pretrained(victim_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(victim_name)

    attack_args = textattack.AttackArgs(
        num_examples=10,
        attack_name=attack_name,
        dataset_name=dataset_name,
        victim_name=victim_name,
        query_budget=100,
        disable_stdout=False,
        parallel=False
    )
    
    attacker = textattack.Attacker(attack, dataset, attack_args)

    attack_results = attacker.attack_dataset()
    print()


if __name__ == '__main__':
    main()