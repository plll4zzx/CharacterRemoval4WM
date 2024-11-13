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
def main():
    # in_file = open("dataset/ag_news/ag_news.json", "r")
    # d1=json.load(in_file)
    # in_file.close()

    data=[("I enjoyed the movie a lot!", 1), ("Absolutely horrible film.", 0), ("Our family had a fun time!", 1)]
    # for d_name in d1:
    #     d2=d1[d_name]
    #     data.append((d2['text'], int(d2['label'])))
        # data.append(((d2['premise'],d2['hypothesis']),int(d2['label'])))
    # print(data)
    # n= 10
    # raw_dataset = load_dataset("snli")
    # dataset=raw_dataset['test']
    # dataset = dataset.shuffle().select(range(n))
    # dd={}
    # with open(f'dataset/snli/snli.txt', 'w') as file:
    #     for idx in range(n):
    #         premise=dataset[idx]['premise']
    #         hypothesis=dataset[idx]['hypothesis']
    #         label=dataset[idx]['label']
    #         dd[idx]={}
    #         dd[idx]['premise']=premise
    #         dd[idx]['hypothesis']=hypothesis
    #         dd[idx]['label']=label
    #         file.write(f"(\"{premise}\",\"{hypothesis}\"),{label})\n") 
    # out_file = open("dataset/snli/snli.json", "w")
    # json.dump(dd, out_file) 
    # out_file.close()

    tokenizer =transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news")
    # tokenizer =transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    # model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
    # tokenizer =transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-snli")
    # model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-snli")
    
    # tokenizer =transformers.AutoTokenizer.from_pretrained("sileod/roberta-base-mnli")
    # model = transformers.AutoModelForSequenceClassification.from_pretrained("sileod/roberta-base-mnli")
    # tokenizer =transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MNLI")
    # model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MNLI")
#    
    # model1 = LSTMForClassification.from_pretrained('lstm-imdb')
    # model1 = LSTMForClassification.from_pretrained('lstm-sst2')
    # model1 = LSTMForClassification.from_pretrained('lstm-ag-news')

    # model1 = WordCNNForClassification.from_pretrained('cnn-sst2')
    # model1 = WordCNNForClassification.from_pretrained('cnn-ag-news')


    # with use_proxy('http://172.18.16.28:7890'):
        # model1 = WordCNNForClassification.from_pretrained('cnn-imdb')

    # print(model1)
    # lstm_model = LSTMForClassification()
    # cnn_model = WordCNNForClassification()
    # tokenizer = lstm_model.tokenizer
    # tokenizer = cnn_model.tokenizer
    # print(tokenizer)

    
    model = transformers.AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    tokenizer = transformers.AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    # model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-SST-2")
    # tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/albert-base-v2-SST-2")
    # model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-imdb")
    # tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/albert-base-v2-imdb")
    # print(tokenizer)
    model_wrapper = textattack.models.wrappers.HuggingFaceEncoderWrapper(model, tokenizer)
    dataset = textattack.datasets.Dataset(data)
    # attack = textattack.attack_sems.TextFoolerJin2019.build(model_wrapper)
    # attack = textattack.attack_sems.BAEGarg2019.build(model_wrapper)
    # attack = textattack.attack_sems.TextBuggerLi2018.build(model_wrapper)
    # attack = textattack.attack_sems.PWWSRen2019.build(model_wrapper)
    # attack = textattack.attack_sems.DeepWordBugGao2018.build(model_wrapper)

    dataset_name = 'mnli'
    attack_name = 'TextFoolerJin2019' #['BAEGarg2019', 'TextBuggerLi2018', 'PWWSRen2019', 'DeepWordBugGao2018']
    victim_name = 'roberta_mismnli'
    attack = getattr(textattack.attack_sems, attack_name).build(model_wrapper)

    # dataset_name = 'imdb'
    # attack_name = 'TextBugger'
    # # attack_name = '512adv_test'
    # victim = 'albert_imdb'

#     # datas7et_name = 'ag_news'
#     # attack_name = 'newTextBugger'
#     # victim = 'bert_ag'，query_budget=1000,
        #,checkpoint_interval=100, checkpoint_dir="./adv_checkpoint/" + dataset_name ,
    attack_args = textattack.AttackArgs(
        num_examples=10,
        attack_name=attack_name,
        dataset_name=dataset_name,
        victim_name=victim_name,
        query_budget=100,
        disable_stdout=False,
        parallel=False
    )#
#     # attack_args = textattack.AttackArgs(num_examples=20, log_to_txt="./test.txt")
    # attacker = textattack.Attacker.from_checkpoint(attack,dataset,checkpoint='adv_checkpoint/imdb/1658885333045.ta.chkpt')
    # attacker.update_attack_args(checkpoint_interval=50)
    attacker = textattack.Attacker(attack, dataset, attack_args)


    # with use_proxy('http://172.18.16.28:7890'):
    attack_results = attacker.attack_dataset()
    # attacker.attack_dataset()
    print()
# #     breakpoint()


if __name__ == '__main__':
    main()

# import subprocess

# output = subprocess.run(['textattack',
#  'attack',
#  '--recipe',
#  'pwws',
#  '--model',
#  'bert-base-uncased-imdb',
#  '--num-examples',
#  '10',
#  '--dataset-from-huggingface',
#  'imdb'], stdout=subprocess.PIPE).stdout.decode('utf-8')
# print(output)