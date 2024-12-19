

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, BertForSequenceClassification, PreTrainedModel
# from torch.optim import AdamW, SGD, Adam
from train_ref_detector import RefDetector
import torch
import transformers
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedModel, PretrainedConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

class MLPBinaryClassifier(PreTrainedModel):
    def __init__(self, config, encoder_path, bert_path):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(encoder_path)
        self.bert = AutoModel.from_pretrained(bert_path)
        
        for param in self.encoder.parameters():
            param.requires_grad = False  # 冻结 encoder 的参数

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            # nn.BatchNorm1d(config.hidden_size // 2),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 2),
            nn.Softmax(dim=1)  # 使用 Softmax 激活函数
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, bert_input_ids=None, bert_attention_mask=None, token_type_ids=None):
        # 获取 encoder 的输出
        with torch.no_grad():
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(encoder_outputs.last_hidden_state.size()).float()
            encoder_hidden_states = torch.sum(encoder_outputs.last_hidden_state*input_mask_expanded, 1)/ torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            # hidden_states = encoder_outputs.last_hidden_state[:, 0, :]  # 通常使用 [CLS] token 的输出
        bert_outputs = self.bert(input_ids=bert_input_ids, attention_mask=bert_attention_mask, token_type_ids=token_type_ids)
        bert_hidden_states = bert_outputs.pooler_output
        
        # 通过 MLP
        hidden_states=torch.cat((encoder_hidden_states, bert_hidden_states), 1)
        logits = self.mlp(hidden_states)
        # logits = self.mlp(bert_hidden_states)

        # 如果提供了 labels，则计算损失
        loss = None
        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
        # return (loss, logits) if loss is not None else logits

class Ref_MLP_Detector(RefDetector):

    def __init__(self, llm_name, wm_name, tokenizer_path="bert-base-uncased", bert_tokenizer_path=None):
        super().__init__(llm_name, wm_name, tokenizer_path)
        self.bert_tokenizer=AutoTokenizer.from_pretrained(bert_tokenizer_path)

    def dataloader_init(self, train_split=0.8, text_len=150):
        def collate_fn(batch):
            list_input_ids = []
            list_attention_mask = []
            list_input_ids_b = []
            list_attention_mask_b = []
            list_token_type_ids=[]
            list_labels=[]
            token_type_flag=False
            
            for i in range(len(batch)):
                tmp_d=self.tokenizer(
                    batch[i]["text"], padding="max_length", truncation=True, 
                    max_length=text_len
                )
                tmp_d_b=self.bert_tokenizer(
                    batch[i]["text"], padding="max_length", truncation=True, 
                    max_length=text_len
                )
                list_input_ids.append(tmp_d['input_ids'])
                list_attention_mask.append(tmp_d['attention_mask'])
                list_labels.append(batch[i]["labels"])
                list_input_ids_b.append(tmp_d_b['input_ids'])
                list_attention_mask_b.append(tmp_d_b['attention_mask'])
                list_token_type_ids.append(tmp_d_b['token_type_ids'])
            
            return {
                "input_ids": torch.tensor(list_input_ids),
                "attention_mask":  torch.tensor(list_attention_mask),
                'bert_input_ids': torch.tensor(list_input_ids_b),
                'bert_attention_mask':  torch.tensor(list_attention_mask_b),
                'token_type_ids': torch.tensor(list_token_type_ids),
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

    
    def train_init(self, encoder_path="bert-base-uncased", bert_path=None, mlp_path='',lr_init=1e-5, hidden_size=768):
        self.train_flag=True

        model_config = PretrainedConfig(hidden_size=hidden_size)
        self.model = MLPBinaryClassifier(model_config, encoder_path, bert_path=bert_path)

        if len(mlp_path)>0:
            self.model.mlp.load_state_dict(torch.load(mlp_path))

        # self.optimizer = AdamW(self.model.parameters(), lr=lr_init, weight_decay=0.8)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr_init)

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.2)

        self.model.to(self.device)


if __name__=='__main__':
    llm_name="facebook/opt-1.3b"
    dataset_name='../../dataset/c4/realnewslike'
    ref_model=Ref_MLP_Detector(
        llm_name=llm_name, wm_name='TS',
        tokenizer_path='sentence-transformers/all-distilroberta-v1',
        bert_tokenizer_path='bert-base-uncased',
    )
    ref_model.load_data(
        dataset_name=dataset_name, data_num=5000, 
        text_len=200
    )
    ref_model.dataloader_init(
        train_split=0.8,
        text_len=200, 
    )
    ref_model.train_init(
        # model_path='saved_model/KGW_.._.._dataset_c4_realnewslike_facebook_opt-1.3b_2024-12-16',
        encoder_path='sentence-transformers/all-distilroberta-v1',
        bert_path='bert-base-uncased',
        lr_init=1e-5, hidden_size=int(768+768)
    )
    # ref_model.froze_layer(f_num=12)
    ref_model.test_epoch()
    ref_model.start_train(num_epochs=10, lr_step=3)
    ref_model.save_model(name='RefMLP')
    