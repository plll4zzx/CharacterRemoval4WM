import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm


model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


dataset = load_dataset("imdb")
train_dataset = dataset["train"]
test_dataset = dataset["test"]


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)


train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)


train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8)


optimizer = AdamW(model.parameters(), lr=2e-5)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.train()

num_epochs = 3
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch in tqdm(train_dataloader):
        inputs = {key: val.to(device) for key, val in batch.items() if key in tokenizer.model_input_names}
        labels = batch["label"].to(device)

        outputs = model(**inputs)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}")


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_dataloader:
        inputs = {key: val.to(device) for key, val in batch.items() if key in tokenizer.model_input_names}
        labels = batch["label"].to(device)

        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy}")