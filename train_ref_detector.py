
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
import argparse
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns

def plot_scatter(true_values, predicted_values, fig_path=None, title='', mid_point=0):
    if len(true_values)>1000:
        sample_indices = np.random.choice(len(true_values), size=1000, replace=False) 
        true_values=true_values[sample_indices]
        predicted_values=predicted_values[sample_indices]
    min_value=min(true_values.min(), predicted_values.min())
    max_value=max(true_values.max(), predicted_values.max())
    

    plt.figure(figsize=(9, 8))
    # xy = np.vstack([true_values, predicted_values])
    # density = gaussian_kde(xy)(xy)
    # plt.scatter(true_values, predicted_values, c=density, cmap="Blues", s=10, alpha=0.6)
    sns.kdeplot(x=true_values, y=predicted_values, cmap="Blues", fill=True)
    # plt.colorbar(label="Density")  
    plt.scatter(true_values, predicted_values, color="black", alpha=0.3, s=10)

    tmp_max=max(abs(max_value-mid_point), abs(min_value-mid_point))
    left_bond=mid_point-tmp_max
    right_bond=mid_point+tmp_max
    plt.plot([left_bond, right_bond], [left_bond, right_bond], color='red', linestyle='--', label="Ideal Line (y=x)")  # 参考对角线

    fontsize=34
    plt.xlabel("Original Watermark Score", fontsize=fontsize)
    plt.ylabel("Reference Watermark Score", fontsize=fontsize)
    # plt.fill_betweenx([-tmp_max, tmp_max], 0, -tmp_max, color='lightgrey', alpha=0.5)
    # plt.fill_betweenx([left_bond, right_bond], 0, tmp_max, where=[False, False], color='lightgrey', alpha=0.5)
    x_fill = np.linspace(left_bond, mid_point, 100)  
    plt.fill_between(x_fill, mid_point, right_bond, color='lightgray', alpha=0.5)
    x_fill = np.linspace(mid_point, right_bond, 100)  
    plt.fill_between(x_fill, left_bond, mid_point, color='lightgray', alpha=0.5)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.rcParams['font.size'] = fontsize

    if len(title)>0:
        plt.title(title, fontsize=fontsize)
    # plt.legend()
    plt.grid(True)

    
    ax = plt.gca()
    # ax.spines['top'].set_linewidth(2)
    # ax.spines['right'].set_linewidth(2)
    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['left'].set_linewidth(2)
    
    ax.set_xlim([left_bond, right_bond])
    ax.set_ylim([left_bond, right_bond])
    # ax.axhline(0, color='black',linewidth=1)
    # ax.axvline(0, color='black',linewidth=1)
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path)
    else:
        plt.show()

def char_adv(sentence, rand_char_rate=0.1):
    tokens = sentence.split()
    edited_sentence = tokens.copy()
    selected_tokens = []
    max_edits=int(len(tokens)*rand_char_rate)
    solution=np.random.choice(len(tokens), max_edits*2, replace=False)

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
        tokenizer=None, 
        # text_len=None, 
        wm_detector=None, stored_flag=False,
        rand_char_rate=0, rand_times=0, wm_threshold=None,
    ):
        if stored_flag:
            self.dataset=load_json(data_path)
            return
        
        self.wm_detector=wm_detector
        self.tokenizer=tokenizer
        # self.text_len=text_len
        tmp_dataset=load_json(data_path)[0:data_num]
        self.wm_dataset=[]
        self.un_dataset=[]
        self.wm_count=0
        self.un_count=0
        for tmp_d in tqdm(tmp_dataset, ncols=100):
            if tmp_d['wm_detect']['is_watermarked']==True:
                tmp_text=tmp_d['wm_text']
                tmp_score=tmp_d['wm_detect']['score']
                tmp_label=tmp_d['wm_detect']['is_watermarked']
                # tmp_text=self.en_de(tmp_text)
                self.add_data(tmp_text, wm_flag=tmp_label, tmp_score=tmp_score)

                for i in range(rand_times):
                    tmp_rand_char_rate = random.uniform(rand_char_rate/2, rand_char_rate)
                    tmp_text0=char_adv(tmp_text, tmp_rand_char_rate)
                    self.add_data(tmp_text0)

            if tmp_d['un_detect']['is_watermarked']==False:
                tmp_text=tmp_d['un_text']
                tmp_score=tmp_d['un_detect']['score']
                tmp_label=tmp_d['un_detect']['is_watermarked']
                # tmp_text=self.en_de(tmp_text)
                self.add_data(tmp_text, wm_flag=tmp_label, tmp_score=tmp_score)

                for i in range(rand_times):
                    tmp_rand_char_rate = random.uniform(rand_char_rate/2, rand_char_rate)
                    tmp_text0=char_adv(tmp_text, tmp_rand_char_rate)
                    self.add_data(tmp_text0)
        
        wm_d_len=len(self.wm_dataset)
        un_d_len=len(self.un_dataset)
        min_d_len=min(wm_d_len, un_d_len)
        random.shuffle(self.wm_dataset)
        random.shuffle(self.un_dataset)
        self.dataset=self.wm_dataset[0:min_d_len]+self.un_dataset[0:min_d_len]
        random.shuffle(self.dataset)
        
        print(wm_d_len, un_d_len)
        if wm_detector is not None:
            if rand_times>0:
                save_json(self.dataset, data_path[0:-5]+'_'+str(rand_char_rate)+'_'+str(rand_times)+'.json')
            else:
                save_json(self.dataset, data_path[0:-5]+'_'+'.json')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def en_de(self, text):
        if self.tokenizer is not None and self.text_len is not None:
            tmp_ids=self.tokenizer.encode(text, add_special_tokens=False)[0:self.text_len]
            text=self.tokenizer.decode(tmp_ids, skip_special_tokens=True)
        return text

    def add_data(self, text, wm_flag=None, tmp_score=None):
        if wm_flag is None:
            wm_rlt=self.wm_detector(text)
            wm_flag=wm_rlt['is_watermarked']
            tmp_score=wm_rlt['score']
        if wm_flag==False:
            tmp_labels=0
            if tmp_score is None:
                tmp_score=0
            self.un_dataset.append({
                'text':text,
                'labels': tmp_labels,
                'score': tmp_score
            })
            self.un_count+=1
        else:
            tmp_labels=1
            if tmp_score is None:
                tmp_score=1
            self.wm_dataset.append({
                'text':text,
                'labels': tmp_labels,
                'score': tmp_score
            })
            self.wm_count+=1

class RefDetector:

    def __init__(self, llm_name, wm_name, tokenizer_path="bert-base-uncased", device=0):
        self.llm_name=llm_name
        self.wm_name=wm_name
        self.device='cuda:'+str(device)
        self.tokenizer_path=tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(wm_name, llm_name)
        self.train_flag=False

    def load_data(
        self, dataset_name, data_num, #text_len=None, 
        rand_char_rate=0, rand_times=0
    ):
        self.rand_times=rand_times
        self.dataset_name=dataset_name
        # if text_len is None:
        #     data_path="saved_data/"+"_".join([self.wm_name, dataset_name.replace('/','_'), self.llm_name.replace('/','_')])+"_5000.json"
        #     self.dataset=WMDataset(data_path, data_num)
        if rand_times>0:
            data_path="saved_data/"+"_".join([self.wm_name, dataset_name.replace('/','_'), self.llm_name.replace('/','_')])+"_5000_"+str(rand_char_rate)+"_"+str(rand_times)+".json"
            if os.path.exists(data_path):
                self.dataset=WMDataset(data_path, data_num, stored_flag=True)
            else:
                data_path="saved_data/"+"_".join([self.wm_name, dataset_name.replace('/','_'), self.llm_name.replace('/','_')])+"_5000.json"
                wm_scheme=LLM_WM(model_name = self.llm_name, device = self.device, wm_name=self.wm_name)
                self.dataset=WMDataset(
                    data_path, data_num, self.tokenizer, 
                    # text_len=text_len, 
                    wm_detector=wm_scheme.detect_wm, 
                    rand_char_rate=rand_char_rate, rand_times=rand_times, 
                    # wm_threshold=wm_scheme.wm_model.utils.config.threshold
                )
        else:
            data_path="saved_data/"+"_".join([self.wm_name, dataset_name.replace('/','_'), self.llm_name.replace('/','_')])+"_5000_"+".json"
            if os.path.exists(data_path):
                self.dataset=WMDataset(data_path, data_num, stored_flag=True)
            else:
                data_path="saved_data/"+"_".join([self.wm_name, dataset_name.replace('/','_'), self.llm_name.replace('/','_')])+"_5000.json"
                wm_scheme=LLM_WM(model_name = self.llm_name, device = self.device, wm_name=self.wm_name)
                self.dataset=WMDataset(
                    data_path, data_num, self.tokenizer, 
                    # text_len=text_len, 
                    wm_detector=wm_scheme.detect_wm
                )
        print(data_path)
        wmscore_list=[tmp_d['score'] for tmp_d in self.dataset]
        self.d_mean = np.mean(wmscore_list)
        self.d_std = np.std(wmscore_list)
        print('mean', self.d_mean, 'std', self.d_std)
        wm_count=sum([1 for tmp_d in self.dataset if tmp_d['labels']==1])
        un_count=sum([1 for tmp_d in self.dataset if tmp_d['labels']==0])

    def train_epoch(self):
        loss_l=[]
        self.model.train()
        for batch in tqdm(self.train_dataloader, ncols=100):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            # logits = outputs.logits
            # labels = batch["labels"]
            # loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = outputs.loss
            loss.backward()
            loss_l.append(loss.item())
            
            # torch.nn.utils.clip_grad_norm_(
            #     self.model.parameters(), max_norm=1, 
            #     # norm_type='inf'
            # )
            
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Layer: {name} | Param before update: {param.data.norm()}")

            self.optimizer.step()

            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Layer: {name} | Param after update: {param.data.norm()}")
            self.optimizer.zero_grad()
            
        print('loss', np.round(np.mean(loss_l), 6))
    
    def test_epoch(self, wm_threshold=None, epoch_num=-1):
        self.model.eval()
        if wm_threshold is not None:
            wm_threshold=(wm_threshold-self.d_mean)/self.d_std
        logits_l=[]
        t_score_l=[]
        
        loss_l=[]
        metric = evaluate.load(os.path.expanduser('metrics/accuracy/accuracy.py'))
        for idx, batch in enumerate(self.train_dataloader):
            if idx == 20:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            loss_l.append(outputs.loss.item())
            logits = outputs.logits
            if wm_threshold is not None:
                predictions = (logits>wm_threshold).float().reshape((-1))
                labels=(batch["labels"]>wm_threshold).float()
                # logits_l.append(logits.item())
                # t_score_l.append(batch["labels"].item())
            else:
                predictions = torch.argmax(logits, dim=-1)
                labels=batch["labels"]
            metric.add_batch(predictions=predictions, references=labels)
        result=metric.compute()
        print('train loss', np.round(np.mean(loss_l),4))
        print('train', result)
        
        metric = evaluate.load(os.path.expanduser('metrics/accuracy/accuracy.py'))
        loss_l=[]
        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            loss_l.append(outputs.loss.item())
            logits = outputs.logits
            if wm_threshold is not None:
                predictions = (logits>wm_threshold).float().reshape((-1))
                labels=(batch["labels"]>wm_threshold).float()
                logits_l.append(logits.reshape((-1)).cpu().numpy())
                t_score_l.append(batch["labels"].reshape((-1)).cpu().numpy())
            else:
                predictions = torch.argmax(logits, dim=-1)
                labels=batch["labels"]
            metric.add_batch(predictions=predictions, references=labels)
        result=metric.compute()
        if wm_threshold is not None:
            t_score=np.hstack(t_score_l)
            p_score=np.hstack(logits_l)
            os.makedirs('plot', exist_ok=True)
            fig_path=os.path.join(
                'plot', '_'.join([
                    self.wm_name,
                    self.dataset_name.replace('/', '_'), 
                    self.llm_name.replace('/', '_'), 
                    self.tokenizer_path.replace('/', '_'),
                    str(self.rand_times),
                    str(epoch_num)
                ])
            )  
            plot_scatter(t_score, p_score, fig_path=fig_path+'.pdf', title=self.wm_name, mid_point=wm_threshold)
            pearson_r = np.corrcoef(t_score, p_score)[0, 1]
            print("pearson_r", pearson_r )
        print('evaluate loss', np.round(np.mean(loss_l),4))
        print('evaluate', result)

    def froze_layer(self, f_num=10):
        for name, param in self.model.bert.named_parameters():
            if "encoder.layer" in name:  
                layer_num = int(name.split(".")[2])  
                if layer_num < f_num:  
                    param.requires_grad = False

        # for name, param in self.model.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")

    def dataloader_init(self, train_split=0.8, text_len=150, batch_size=16):
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
                list_labels.append(
                    float((batch[i]["score"]-self.d_mean)/self.d_std)
                )
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
            self.dataset[0:train_num], shuffle=True, batch_size=batch_size, 
            collate_fn=collate_fn
        )
        self.eval_dataloader = DataLoader(
            self.dataset[train_num+1:], shuffle=True, batch_size=batch_size, 
            collate_fn=collate_fn
        )

    def train_init(self, model_path=None, lr_init=1e-5, gamma=0.2):
        if model_path is None:
            model_path="bert-base-uncased"
        self.train_flag=True

        self.model_path=model_path
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=1, 
            problem_type="regression", 
            torch_dtype="auto"
        )
        self.model.to(self.device)

        # self.optimizer = AdamW(self.model.parameters(), lr=lr_init, weight_decay=0.8)
        self.optimizer = Adam(self.model.parameters(), lr=lr_init)
        # self.optimizer = SGD(self.model.parameters(), lr=lr_init, momentum=0.9)

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=gamma)


    def start_train(self, num_epochs=3, lr_step=8, wm_threshold=None):
        # loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            if epoch%lr_step==0 and epoch>0:
                self.lr_scheduler.step()
                print('lr', self.optimizer.param_groups[0]['lr'])
            print('epoch:', epoch)
            self.train_epoch()
            self.test_epoch(wm_threshold=wm_threshold, epoch_num=epoch)
    
    def save_model(self, model_path):
        # model_path=os.path.join(
        #     'saved_model', '_'.join([
        #         name,
        #         self.wm_name,
        #         self.dataset_name.replace('/', '_'), 
        #         self.llm_name.replace('/', '_'), 
        #         self.tokenizer_path.replace('/', '_'),
        #         str(datetime.datetime.now().date())
        #     ])
        # )    
        self.model.save_pretrained(model_path)
        # torch.save(self.model.state_dict(), model_path+".pth")


if __name__=='__main__':
    # llm_name="facebook/opt-1.3b"#"../model/Llama3.1-8B_hg"
    dataset_name='../../dataset/c4/realnewslike'
    tokenizer_path='bert-base-uncased'
    # python train_ref_detector.py --wm_name "Unbiased" --num_epochs 15 --rand_char_rate 0.1
    parser = argparse.ArgumentParser(description='train ref detector')
    parser.add_argument('--llm_name', type=str, default='facebook/opt-1.3b')
    parser.add_argument('--wm_name', type=str, default='UPV')
    parser.add_argument('--num_epochs', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--rand_char_rate', type=float, default=0.15)
    parser.add_argument('--lr_step', type=int, default=3)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument(
        '--model_path', type=str, 
        default=None,
        # default="saved_model/RefDetector_UPV_.._.._dataset_c4_realnewslike_facebook_opt-1.3b_bert-base-uncased_2025-02-16"
    )
    parser.add_argument('--ths', type=float, default=0.5)
    parser.add_argument('--rand_times', type=int, default=5)
    
    args = parser.parse_args()
    ref_model=RefDetector(
        llm_name=args.llm_name, 
        wm_name=args.wm_name, 
        tokenizer_path=tokenizer_path,
        device=args.device,
    )
    ref_model.load_data(
        dataset_name=dataset_name, data_num=5000, 
        rand_char_rate=args.rand_char_rate, rand_times=args.rand_times
        # text_len=150,
    )
    ref_model.dataloader_init(
        train_split=0.8,
        text_len=150, 
        batch_size=16
    )
    ref_model.train_init(
        model_path=args.model_path,
        lr_init=args.lr, gamma=0.5
    )
    # ref_model.froze_layer(f_num=12)
    ref_model.test_epoch(wm_threshold=args.ths)
    ref_model.start_train(num_epochs=args.num_epochs, lr_step=args.lr_step, wm_threshold=args.ths)
    if args.num_epochs>0:
        
        model_path=os.path.join(
            'saved_model', '_'.join([
                'RefDetector',
                args.wm_name,
                dataset_name.replace('/', '_'), 
                args.llm_name.replace('/', '_'), 
                tokenizer_path.replace('/', '_'),
                str(args.rand_char_rate),
                str(args.rand_times),
            ])
        )   
        ref_model.save_model(model_path=model_path)