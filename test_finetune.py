import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from textattack.utils import load_json, save_json
from llm_wm import LLM_WM
from read_data import c4
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

class WmDataset(Dataset):
    def __init__(self, tokenizer, target_llm_name, llm_name, wm_name, max_length=512):
        dataset_name='../../dataset/c4/realnewslike'
        self.tokenizer = tokenizer
        self.tokenizer.padding_side='left'
        self.max_length = max_length
        self.data = load_json(
            "saved_data/"+"_".join([wm_name, dataset_name.replace('/','_'), target_llm_name.replace('/','_')])+"_5000.json"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['wm_text']
        encoding = self.tokenizer(
            text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)  # Remove batch dimension
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}

class Learned_WM_Model:

    def __init__(self, llm_name, wm_name, target_llm_name):
        self.llm_name=llm_name
        self.wm_name=wm_name
        self.target_llm_name=target_llm_name
        self.device='cuda'
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
        print(wm_name, llm_name)
        self.train_flag=False
    
    def load_dataset(
        self, 
        batch_size=8,
        eval_text_len=50,
        eval_file_num=10,
        eval_file_data_num=10,
        eval_dataset_name='../../dataset/c4/realnewslike',
        eval_rand_seed=123,
    ):
        self.dataset = WmDataset(self.tokenizer, self.target_llm_name, self.llm_name, self.wm_name)
        # ========== Step 4: Split Dataset into Train and Test ==========
        train_size = int(0.8 * len(self.dataset))  # 80% for training
        test_size = len(self.dataset) - train_size  # 20% for testing

        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.eval_dataset = c4(dir_path=eval_dataset_name, file_num=eval_file_num, file_data_num=eval_file_data_num, rand_seed=eval_rand_seed)
        self.eval_dataset.load_data(eval_text_len)
    
    def load_wm_detector(self):
        wm_scheme=LLM_WM(model_name = self.target_llm_name, device = "cuda", wm_name=self.wm_name)
        self.wm_detector=wm_scheme.detect_wm

    def train_init(self, gamma=0.2):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_name,
            load_in_4bit=True,  # Change to `load_in_4bit=True` for 4-bit mode
            device_map="auto"
        )

        # Move model to GPU if available
        # self.model.to(self.device)
        self.model = prepare_model_for_kbit_training(self.model)
        lora_config = LoraConfig(
            r=16,  # LoRA rank (adjust based on memory)
            lora_alpha=32,  # Scaling factor
            target_modules=["q_proj", "v_proj"],  # LoRA applied to attention layers
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"  # Required for causal language models like OPT
        )

        # Wrap model with LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=gamma)
    
    
    def start_train(self, num_epochs=3, lr_step=8, gradient_accumulation_steps = 4):
        # loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            if epoch%lr_step==0 and epoch>0:
                self.lr_scheduler.step()
                print('lr', self.optimizer.param_groups[0]['lr'])
            print('epoch:', epoch)
            tmp_train_loss=self.train_epoch(epoch, gradient_accumulation_steps)
            print(f"Epoch {epoch + 1} completed. Average Loss: {tmp_train_loss:.4f}")
            self.test_epoch()
            
    def save_model(self):
        self.model.save_pretrained("saved_model/opt_finetuned")

    def train_epoch(self, epoch, gradient_accumulation_steps = 4):
        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps  # Normalize loss for accumulation
            loss.backward()  # Compute gradients

            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                self.optimizer.step()  # Update model parameters
                self.optimizer.zero_grad()  # Reset gradients

            total_loss += loss.item() * gradient_accumulation_steps

            if step % 10 == 0:  # Print every 10 steps
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")
        return total_loss / len(self.train_loader)
    
    def test_epoch(self):
        self.model.eval()

        test_loss = 0

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                test_loss += loss.item()

        print(f"Test Loss: {test_loss / len(self.test_loader):.4f}")

        wm_rate=0
        batch_size = 4  # Adjust based on GPU memory
        num_batches = (self.eval_dataset.data_num + batch_size - 1) // batch_size  # Compute number of batches

        for i in range(num_batches):
            batch_prompts = [
                p[0]
                for p in self.eval_dataset.data[i * batch_size: (i + 1) * batch_size]
            ]
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=50).to(self.device)  # Batch encoding

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=150,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    return_dict_in_generate=True,  # Required to get hidden states
                    output_hidden_states=True  # Returns hidden states of all tokens
                )
            # Extract generated token IDs
            output_ids = outputs.sequences

            # Extract hidden states of each generated token
            hidden_states = output_ids.hidden_states  # For decoder models (OPT, LLaMA)

            # Convert hidden states to tensor (Shape: [num_layers, batch_size, seq_len, hidden_dim])
            hidden_states_tensor = torch.stack(hidden_states)  # Stacking across layers

            generated_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for j in range(len(generated_texts)):
                wm_rlt=self.wm_detector(generated_texts[j])
                if wm_rlt['is_watermarked']==True:
                    wm_rate+=1
        wm_rate=wm_rate/self.eval_dataset.data_num
        print('wm_rate:', wm_rate)

if __name__=="__main__":
    llm_name='facebook/opt-125m'
    wm_name='KGW'
    target_llm_name='facebook/opt-1.3b'
    ld_model=Learned_WM_Model(llm_name, wm_name, target_llm_name)
    ld_model.load_dataset(
        batch_size=4,
        eval_text_len=50,
        eval_file_num=10,
        eval_file_data_num=10,
        eval_dataset_name='../../dataset/c4/realnewslike',
        eval_rand_seed=123,
    )
    ld_model.load_wm_detector()
    ld_model.train_init(
        gamma=0.2
    )
    ld_model.test_epoch()
    ld_model.start_train(
        num_epochs=3,
        lr_step=8,
        gradient_accumulation_steps = 1
    )
    # ld_model.save_model()

        