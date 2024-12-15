
from textattack.utils import load_json
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from transformers import get_scheduler
import torch
from accelerate.test_utils.testing import get_backend
from tqdm.auto import tqdm
import evaluate
from torch.utils.data import Dataset
import numpy as np
import os

class WMDataset(Dataset):
    def __init__(self, data_path, data_num, tokenizer=None, text_len=150):
        tmp_dataset=load_json(data_path)[0:data_num]
        self.dataset=[]
        for tmp_d in tmp_dataset:
            if tmp_d['wm_detect']['is_watermarked']==True:
                tmp_text=tmp_d['wm_text']
                if tokenizer is not None:
                    tmp_ids=tokenizer.encode(tmp_text)[1:text_len]
                    tmp_text=tokenizer.decode(tmp_ids)
                self.dataset.append({
                    'text':tmp_text,
                    'labels': 1,
                })
            if tmp_d['un_detect']['is_watermarked']==False:
                tmp_text=tmp_d['un_text']
                if tokenizer is not None:
                    tmp_ids=tokenizer.encode(tmp_text)[1:text_len]
                    tmp_text=tokenizer.decode(tmp_ids)
                self.dataset.append({
                    'text':tmp_text,
                    'labels': 0,
                })

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

    def load_data(self, dataset_name, data_num, text_len=150):
        self.dataset_name=dataset_name
        data_path="saved_data/"+"_".join([self.wm_name, dataset_name.replace('/','_'), self.llm_name.replace('/','_')])+"_5000.json"
        self.dataset=WMDataset(data_path, data_num, self.tokenizer, text_len=text_len)

    def train_init(self, num_epochs=3, train_split=0.9, text_len=150):
        
        def collate_fn(batch):
            list_input_ids = []
            list_attention_mask = []
            list_token_type_ids=[]
            list_labels=[]
            
            for i in range(len(batch)):
                tmp_d=self.tokenizer(
                    batch[i]["text"], padding="max_length", truncation=True, 
                    max_length=text_len
                )
                list_input_ids.append(tmp_d['input_ids'])
                list_attention_mask.append(tmp_d['attention_mask'])
                list_token_type_ids.append(tmp_d['token_type_ids'])
                list_labels.append(batch[i]["labels"])
            
            return {
                "input_ids": torch.tensor(list_input_ids),
                "attention_mask":  torch.tensor(list_attention_mask),
                'token_type_ids': torch.tensor(list_token_type_ids),
                'labels': torch.tensor(list_labels)
            }

        train_num=round(self.dataset.__len__()*train_split)
        train_dataloader = DataLoader(
            self.dataset[0:train_num], shuffle=True, batch_size=8, 
            collate_fn=collate_fn
        )
        eval_dataloader = DataLoader(
            self.dataset[train_num+1:], shuffle=True, batch_size=8, 
            collate_fn=collate_fn
        )

        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, torch_dtype="auto")

        optimizer = AdamW(self.model.parameters(), lr=1e-5)

        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)
        # lr_scheduler = get_scheduler(
        #     name="linear", 
        #     optimizer=optimizer, 
        #     num_warmup_steps=0, 
        #     num_training_steps=num_training_steps
        # )

        self.model.to(self.device)

        # loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            print('epoch:', epoch)
            # progress_bar = range(len(train_dataloader)))
            loss_l=[]
            self.model.train()
            for batch in tqdm(train_dataloader, ncols=100):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                logits = outputs.logits
                # labels = batch["labels"]
                # loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = outputs.loss
                loss.backward()
                loss_l.append(loss.item())

                optimizer.step()
                optimizer.zero_grad()
                
            print('loss', np.round(np.mean(loss_l),4))
            if epoch%5==0 and epoch>0:
                lr_scheduler.step()
                print('lr', optimizer.param_groups[0]['lr'])
            
            metric = evaluate.load("accuracy")
            self.model.eval()
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])

            result=metric.compute()
            print(result)
    
    def save_model(self):
        model_path=os.path.join(
            'saved_model', '_'.join([
                self.wm_name,
                self.dataset_name.replace('/','_'), 
                self.llm_name.replace('/','_'), 
            ])
        )    
        self.model.save_pretrained(model_path)
        # torch.save(self.model.state_dict(), model_path+".pth")



if __name__=='__main__':
    llm_name="facebook/opt-1.3b"
    dataset_name='../../dataset/c4/realnewslike'
    ref_model=RefDetector(llm_name=llm_name, wm_name='SemStamp')
    ref_model.load_data(dataset_name=dataset_name, data_num=2000, text_len=100)
    ref_model.train_init(num_epochs=10, text_len=100)
    ref_model.save_model()