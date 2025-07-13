import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, BloomForCausalLM, GPT2LMHeadModel, LlamaForSequenceClassification, LlamaTokenizer
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig 
import loralib as lora
import pandas as pd
from collections import Counter
from torch.utils.data import random_split
from tqdm import tqdm



class all_dataset(Dataset):
    def __init__(self, file_path, type):
        self.tokenizer = LlamaTokenizer.from_pretrained('ziqingyang/chinese-llama-plus-lora-7b', add_eos_token=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.sentence_token = []
        self.label_token = []
        self.type = type
        self.data = pd.read_csv(file_path)

        self.sentence_list = [sentence for sentence in self.data['sentence'].values]
        
        if self.type == 'train':
            self.label_list = [str(label) for label in self.data['label_for_kaggle'].values]
            self.label_token = [torch.tensor(i) for i in self.tokenizer.encode(self.label_list)]
            label_count = Counter(self.label_list)
            print("Total data size is %0d" % len(self.data), f"and {label_count}")
        print(self.tokenizer.encode(self.sentence_list))
        exit()
        self.sentence_token = [torch.tensor(i) for i in self.tokenizer.encode(self.sentence_list, padding=True)]
        

        # all_len = torch.tensor([len(i) for i in self.sentence_token.float()])
        # self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        tokens = self.sentence_token[item]
        print(tokens)
        if self.type == 'train':
            label = self.label_token[item]
        
        # padding = self.max_seq_len - tokens.shape[0]
        # if padding > 0:
        #     tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64)))
        #     self.sentence_token[item] = tokens
        # elif padding < 0:
        #     tokens = tokens[:self.max_seq_len]
        #     self.sentence_token[item] = tokens

        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()


        print(mask)
        print(tokens)
        exit()
        if self.type == 'train':
            return tokens, label, mask
        else:
            return tokens, mask
        

epochs = 200
bs = 1

train_file_path = './COS_train.csv'
test_file_path = './COS_test.csv'
train_dataset, val_dataset = random_split(all_dataset(train_file_path, 'train'), [0.8, 0.2], generator=torch.Generator().manual_seed(42))
test_dataset = all_dataset(test_file_path, 'test')
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True, drop_last=True)

model = LlamaForSequenceClassification.from_pretrained('ziqingyang/chinese-llama-plus-lora-7b', device_map='auto', load_in_8bit=True, )
device = torch.device('cuda:0')

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    fan_in_fan_out=True,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
model = model.to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=20, num_training_steps=epochs * len(train_dataloader)
    )
# save_config(args)
step = 0
no_progress_step = 0
best_loss = float('inf')
tf_logger = SummaryWriter(log_dir=f'log/gpt2')

for epoch in range(epochs):
        print(f">>> Epochs {epoch + 1}")
        train_losses = []
        progress = tqdm(total=len(train_dataloader), desc='training_progress')

        for batch_idx, (sentence, label) in enumerate(train_dataloader):
            model.zero_grad()
            sentence, label = sentence.to(device), label.to(device)

            output = model(input_embeds=model.transformer.wte(sentence))
            print(output)
            exit()
            loss = criterion(scores, labels)

            loss.backward()
            optimizer.step()
            step_lr_scheduler.step()
            optimizer.zero_grad()

            

            train_losses.append(loss.item())
            tf_logger.add_scalar('training loss', loss.item(), step)
            step += 1
            progress.set_postfix({
                'loss': np.mean(train_losses),
                'lr': step_lr_scheduler.optimizer.param_groups[0]['lr'],
                })
            progress.update()
        progress.close()

        #validation
        model.eval()
        progress = tqdm(total=len(val_dataloader), desc='validation_progress')
        val_losses = []
        for batch_idx, (images, labels) in enumerate(val_dataloader):

            images = images.to(device)
            labels = labels.to(device)

            scores = model(images)
            loss = criterion(scores, labels)

            val_losses.append(loss.item())
            tf_logger.add_scalar('validation loss', loss.item(), step)
            step += 1
            progress.set_postfix({
                'val_loss': np.mean(val_losses),
                })
            progress.update()
        tf_logger.add_scalar('validation loss', np.mean(val_losses), epoch)
        progress.close()

        if np.mean(val_losses) < best_loss:
            no_progress_step = 0 #重置

            torch.save(
                model,
                "./models/CNN_caption_type_classifier.pt"  
            )
            print(f">>> model saved, epoch={epoch}, val_loss={np.mean(val_losses)}")

            best_loss = np.mean(val_losses)
        else:
            print(f">>> val_loss={np.mean(val_losses)} > {best_loss}, don't save the model")
            no_progress_step += 1
            print(f">>> The loss accumulate {no_progress_step} times have no progress")
            if no_progress_step == 30:
                break