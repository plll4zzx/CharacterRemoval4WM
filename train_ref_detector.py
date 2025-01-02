
from textattack.utils import load_json, save_json, find_homo
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, BertForSequenceClassification
from torch.optim import AdamW, SGD, Adam
from torch.utils.data import Dataset, DataLoader
import torch
from accelerate.test_utils.testing import get_backend
from tqdm import tqdm
import evaluate
import numpy as np
import os
import datetime
import random
from llm_wm import LLM_WM

def char_adv(sentence, rand_char_rate=0.1):
    tokens = sentence.split()
    edited_sentence = tokens.copy()
    selected_tokens = []
    max_edits=int(len(tokens)*rand_char_rate)
    solution=np.random.choice(len(tokens), max_edits*2)

    for loc in solution:
        if  len(selected_tokens) < max_edits:  

            half_token_len=len(edited_sentence[loc])//2
            if half_token_len<=1:
                continue

            selected_tokens.append(loc)

            operation = 2 

            tmp_token=edited_sentence[loc]

            if operation == 1:  # Delete
                edited_sentence[loc] = tmp_token[:half_token_len] + tmp_token[half_token_len+1:]
            elif operation == 2:  # Replace
                tmp_char=tmp_token[half_token_len]
                edited_sentence[loc] = tmp_token[:half_token_len] +find_homo(tmp_char)+ tmp_token[half_token_len+1:] 
            elif operation == 3:  # Insert
                edited_sentence[loc] = tmp_token[:half_token_len] +'@'+ tmp_token[half_token_len:]

    # Reconstruct sentence
    modified_sentence = " ".join(edited_sentence)
    return modified_sentence

class WMDataset(Dataset):
    def __init__(
        self, data_path, data_num, 
        tokenizer=None, text_len=None, wm_detector=None, stored_flag=False,
        rand_char_rate=0, rand_times=0,
    ):
        if stored_flag:
            self.dataset=load_json(data_path)
            return
        
        tmp_dataset=load_json(data_path)[0:data_num]
        self.dataset=[]
        count_un=0
        count_wm=0
        for tmp_d in tqdm(tmp_dataset, ncols=100):
            if tmp_d['wm_detect']['is_watermarked']==True:
                tmp_text=tmp_d['wm_text']
                tmp_labels=1
                if tokenizer is not None and text_len is not None:
                    tmp_ids=tokenizer.encode(tmp_text, add_special_tokens=False)[0:text_len]
                    tmp_text=tokenizer.decode(tmp_ids, skip_special_tokens=True)
                    wm_rlt=wm_detector(tmp_text)
                    wm_flag=wm_rlt['is_watermarked']
                    if wm_flag==False:
                        count_un+=1
                        tmp_labels=0
                    else:
                        count_wm+=1
                        tmp_labels=1
                    self.dataset.append({
                        'text':tmp_text,
                        'labels': tmp_labels,
                        'score': wm_rlt['score']
                    })
                    if wm_rlt['score']>=5 and rand_times>0:
                        for i in range(rand_times):
                            tmp_text0=char_adv(tmp_text, rand_char_rate)
                            wm_rlt0=wm_detector(tmp_text0)
                            wm_flag0=wm_rlt0['is_watermarked']
                            if wm_flag0==False:
                                count_un+=1
                                tmp_labels0=0
                            else:
                                count_wm+=1
                                tmp_labels0=1
                            self.dataset.append({
                                'text': tmp_text0,
                                'labels': tmp_labels0,
                                'score': wm_rlt0['score']
                            })
            if tmp_d['un_detect']['is_watermarked']==False:
                tmp_text=tmp_d['un_text']
                if tokenizer is not None and text_len is not None:
                    tmp_ids=tokenizer.encode(tmp_text, add_special_tokens=False)[0:text_len]
                    tmp_text=tokenizer.decode(tmp_ids, skip_special_tokens=True)
                    wm_rlt=wm_detector(tmp_text)
                    self.dataset.append({
                        'text':tmp_text,
                        'labels': 0,
                        'score': wm_rlt['score']
                    })
        random.shuffle(self.dataset)
        print(count_wm, count_un)
        if wm_detector is not None:
            if rand_times>0:
                save_json(self.dataset, data_path[0:-5]+'_'+str(text_len)+'_'+str(rand_char_rate)+'_'+str(rand_times)+'.json')
            else:
                save_json(self.dataset, data_path[0:-5]+'_'+str(text_len)+'.json')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class RefDetector:

    def __init__(self, llm_name, wm_name, tokenizer_path="bert-base-uncased"):
        self.llm_name=llm_name
        self.wm_name=wm_name
        self.device='cuda'
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(wm_name, llm_name)
        self.train_flag=False

    def load_data(
        self, dataset_name, data_num, text_len=None, 
        rand_char_rate=0, rand_times=0
    ):
        self.dataset_name=dataset_name
        if text_len is None:
            data_path="saved_data/"+"_".join([self.wm_name, dataset_name.replace('/','_'), self.llm_name.replace('/','_')])+"_5000.json"
            self.dataset=WMDataset(data_path, data_num)
        elif rand_times>0:
            data_path="saved_data/"+"_".join([self.wm_name, dataset_name.replace('/','_'), self.llm_name.replace('/','_')])+"_5000_"+str(text_len)+"_"+str(rand_char_rate)+"_"+str(rand_times)+".json"
            if os.path.exists(data_path):
                self.dataset=WMDataset(data_path, data_num, stored_flag=True)
            else:
                data_path="saved_data/"+"_".join([self.wm_name, dataset_name.replace('/','_'), self.llm_name.replace('/','_')])+"_5000.json"
                wm_scheme=LLM_WM(model_name = self.llm_name, device = "cuda", wm_name=self.wm_name)
                self.dataset=WMDataset(data_path, data_num, self.tokenizer, text_len=text_len, wm_detector=wm_scheme.detect_wm, rand_char_rate=rand_char_rate, rand_times=rand_times)
        else:
            data_path="saved_data/"+"_".join([self.wm_name, dataset_name.replace('/','_'), self.llm_name.replace('/','_')])+"_5000_"+str(text_len)+".json"
            if os.path.exists(data_path):
                self.dataset=WMDataset(data_path, data_num, stored_flag=True)
            else:
                data_path="saved_data/"+"_".join([self.wm_name, dataset_name.replace('/','_'), self.llm_name.replace('/','_')])+"_5000.json"
                wm_scheme=LLM_WM(model_name = self.llm_name, device = "cuda", wm_name=self.wm_name)
                self.dataset=WMDataset(data_path, data_num, self.tokenizer, text_len=text_len, wm_detector=wm_scheme.detect_wm)

    def train_epoch(self):
        loss_l=[]
        self.model.train()
        for batch in tqdm(self.train_dataloader, ncols=100):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            logits = outputs.logits
            # labels = batch["labels"]
            # loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = outputs.loss
            loss.backward()
            loss_l.append(loss.item())

            self.optimizer.step()
            self.optimizer.zero_grad()
            
        print('loss', np.round(np.mean(loss_l), 6))
    
    def test_epoch(self):
        self.model.eval()

        loss_l=[]
        metric = evaluate.load('/home/plll/python/metrics/accuracy/accuracy.py')
        for idx, batch in enumerate(self.train_dataloader):
            if idx == 20:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            loss_l.append(outputs.loss.item())
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        result=metric.compute()
        print('train loss', np.round(np.mean(loss_l),4))
        print('train', result)
        
        metric = evaluate.load('/home/plll/python/metrics/accuracy/accuracy.py')
        loss_l=[]
        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            loss_l.append(outputs.loss.item())
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        result=metric.compute()
        print('evaluate loss', np.round(np.mean(loss_l),4))
        print('evaluate', result)

    def froze_layer(self, f_num=10):
        for name, param in self.model.bert.named_parameters():
            if "encoder.layer" in name:  # 只对 encoder 层操作
                layer_num = int(name.split(".")[2])  # 提取层号
                if layer_num < f_num:  # 冻结前 10 层
                    param.requires_grad = False

        # 验证冻结结果
        # for name, param in self.model.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")

    def dataloader_init(self, train_split=0.8, text_len=150):
        def collate_fn(batch):
            list_input_ids = []
            list_attention_mask = []
            list_token_type_ids=[]
            list_labels=[]
            token_type_flag=False
            
            for i in range(len(batch)):
                tmp_d=self.tokenizer(
                    batch[i]["text"], padding="max_length", truncation=True, 
                    max_length=text_len
                )
                list_input_ids.append(tmp_d['input_ids'])
                list_attention_mask.append(tmp_d['attention_mask'])
                list_labels.append(batch[i]["labels"])
                if 'token_type_ids' in tmp_d:
                    token_type_flag=True
                    list_token_type_ids.append(tmp_d['token_type_ids'])
            
            if token_type_flag:
                return {
                    "input_ids": torch.tensor(list_input_ids),
                    "attention_mask":  torch.tensor(list_attention_mask),
                    'token_type_ids': torch.tensor(list_token_type_ids),
                    'labels': torch.tensor(list_labels)
                }
            else:
                return {
                    "input_ids": torch.tensor(list_input_ids),
                    "attention_mask":  torch.tensor(list_attention_mask),
                    'labels': torch.tensor(list_labels)
                }


        train_num=round(self.dataset.__len__()*train_split)
        self.train_dataloader = DataLoader(
            self.dataset[0:train_num], shuffle=True, batch_size=16, 
            collate_fn=collate_fn
        )
        self.eval_dataloader = DataLoader(
            self.dataset[train_num+1:], shuffle=True, batch_size=16, 
            collate_fn=collate_fn
        )

    def train_init(self, model_path="bert-base-uncased", lr_init=1e-5, gamma=0.2):
        self.train_flag=True

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=2, torch_dtype="auto"
        )
        self.model.to(self.device)

        # self.optimizer = AdamW(self.model.parameters(), lr=lr_init, weight_decay=0.8)
        self.optimizer = Adam(self.model.parameters(), lr=lr_init)

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=gamma)


    def start_train(self, num_epochs=3, lr_step=8):
        # loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            if epoch%lr_step==0 and epoch>0:
                self.lr_scheduler.step()
                print('lr', self.optimizer.param_groups[0]['lr'])
            print('epoch:', epoch)
            self.train_epoch()
            self.test_epoch()
    
    def save_model(self, name='RefDetector'):
        model_path=os.path.join(
            'saved_model', '_'.join([
                name,
                self.wm_name,
                self.dataset_name.replace('/', '_'), 
                self.llm_name.replace('/', '_'), 
                str(datetime.datetime.now().date())
            ])
        )    
        self.model.save_pretrained(model_path)
        # torch.save(self.model.state_dict(), model_path+".pth")


if __name__=='__main__':
    llm_name="facebook/opt-1.3b"
    dataset_name='../../dataset/c4/realnewslike'
    ref_model=RefDetector(llm_name=llm_name, wm_name='KGW')
    ref_model.load_data(
        dataset_name=dataset_name, data_num=5000, 
        text_len=100,
        rand_char_rate=0.05, rand_times=5
    )
    ref_model.dataloader_init(
        train_split=0.8,
        text_len=100, 
    )
    ref_model.train_init(
        # model_path='saved_model/KGW_.._.._dataset_c4_realnewslike_facebook_opt-1.3b_2024-12-16',
        lr_init=5e-5
    )
    # ref_model.froze_layer(f_num=12)
    ref_model.test_epoch()
    ref_model.start_train(num_epochs=10, lr_step=5)
    ref_model.save_model(name='RefDetector')