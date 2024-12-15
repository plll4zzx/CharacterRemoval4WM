from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
import torch
from accelerate.test_utils.testing import get_backend
from tqdm.auto import tqdm
import evaluate
from transformers import DataCollatorWithPadding

dataset = load_dataset("sst")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, torch_dtype="auto")

# def tokenize_function(examples):
#     encoded = tokenizer(
#         examples["sentence"], padding="max_length", truncation=True
#     )
#     return {key: torch.tensor(val) for key, val in encoded.items()}
# tokenized_datasets = dataset.map(tokenize_function, batched=True)

# tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "tokens", "tree"])
# tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))

# train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
# eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["sentence", "tokens", "tree"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
def collate_fn(batch):
    list_input_ids = []
    list_attention_mask = []
    list_token_type_ids=[]
    
    for i in range(len(batch)):
        list_input_ids.append(batch[i]['input_ids'])
        list_attention_mask.append(batch[i]['attention_mask'])
        list_token_type_ids.append(batch[i]['token_type_ids'])
    
    return {
        "input_ids": torch.tensor(list_input_ids),
        "attention_mask":  torch.tensor(list_attention_mask),
        'token_type_ids': torch.tensor(list_token_type_ids)
    }

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, 
    collate_fn=collate_fn
)
eval_dataloader = DataLoader(
    tokenized_datasets["test"], shuffle=True, batch_size=8, 
    collate_fn=collate_fn
)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
model.to(device)

progress_bar = tqdm(range(num_training_steps))

def trans_batch(batch_data, device):
    new_data={}
    for k, v in batch_data.items():
        if isinstance(v, list):
            new_data[k]=torch.vstack(v).to(device)
        else:
            new_data[k]=v.to(device)
    return new_data

model.train()
loss_fn = torch.nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # batch=trans_batch(batch, device)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        labels = batch["labels"]
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        # loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()