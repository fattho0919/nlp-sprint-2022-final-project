import torch
from sklearn.metrics import f1_score
from torch.nn import BCEWithLogitsLoss
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, BloomForSequenceClassification, BitsAndBytesConfig
import bitsandbytes as bnb
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig 
import loralib as lora
import pandas as pd
from collections import Counter
from torch.utils.data import random_split
from tqdm import tqdm
from zhon.hanzi import punctuation

punctuation += '.()'
class all_dataset(Dataset):
    def __init__(self, file_path, type, tokenizer):
        self.tokenizer = tokenizer
        self.sentence_token = []
        self.type = type
        self.data = pd.read_csv(file_path)
        self.sentence_list = [sentence for sentence in self.data['sentence'].values]
        for i in range(len(self.sentence_list)):
            for j in punctuation:
                self.sentence_list[i] = self.sentence_list[i].replace(j, '')
            self.sentence_list[i] = '<s>' + self.sentence_list[i] + '</s>'
        
        if self.type == 'train':
            self.label_list = [label for label in self.data['label_for_kaggle'].values]
            label_count = Counter(self.label_list)
            print("Total data size is %0d" % len(self.data), f"and {label_count}")

        self.sentence_token = self.tokenizer(self.sentence_list, padding=True, return_tensors='pt')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        tokens = self.sentence_token[item].ids
        mask = self.sentence_token[item].attention_mask

        c = 0
        for i in tokens:
            if i == 3:
                c+=1
        tokens = tokens[c:]
        for i in range(c):
            tokens.append(3)

        c = 0
        for i in mask:
            if i == 0:
                c+=1
        mask = mask[c:]
        for i in range(c):
            mask.append(0)
        
        tokens = torch.tensor(tokens)
        mask = torch.tensor(mask)

        if self.type == 'train':
            label = torch.tensor(self.label_list[item])
            label = torch.eye(2)[label]
            return tokens, mask, label
        else:
            return tokens, mask
        
#model & tokenizer setting
lr = 1e-5
epochs = 40
bs = 2
model_type = 'ckip-joint/bloom-3b-zh'
bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
tokenizer = AutoTokenizer.from_pretrained(model_type)
model = BloomForSequenceClassification.from_pretrained(model_type, num_labels=2, quantization_config=bnb_config, device_map={"":0})
device = torch.device('cuda:0')
model.train()

#dataset setting
train_file_path = './COS_train.csv'
test_file_path = './COS_test.csv'
train_dataset, val_dataset = random_split(all_dataset(train_file_path, 'train', tokenizer), [0.8, 0.2], generator=torch.Generator().manual_seed(42))
test_dataset = all_dataset(test_file_path, 'test', tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True, drop_last=True)

loss_function = BCEWithLogitsLoss()
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=200, num_training_steps=epochs * len(train_dataloader)
    )

step = 0
no_progress_step = 0
best_loss = float('inf')
best_f1 = 0
tf_logger = SummaryWriter(log_dir=f"log/{model_type.split('/')[-1]}")

for epoch in range(epochs):
        print(f">>> Epochs {epoch + 1}")
        train_losses = []
        progress = tqdm(total=len(train_dataloader), desc='training_progress')

        #training
        for batch_idx, (sentence, mask, label) in enumerate(train_dataloader):
            model.zero_grad()
            sentence, mask, label = sentence.to(device),mask.to(device), label.to(device)

            output = model(input_ids=sentence, attention_mask=mask).logits
            loss = loss_function(output, label)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            train_losses.append(loss.item())
            tf_logger.add_scalar('training loss', loss.item(), step)
            step += 1
            progress.set_postfix({
                'loss': np.mean(train_losses),
                'lr': scheduler.optimizer.param_groups[0]['lr'],
                })
            progress.update()
        progress.close()

        #validation
        with torch.no_grad():
            progress = tqdm(total=len(val_dataloader), desc='validation_progress')
            val_losses = []
            output_list = []
            label_list = []
            for batch_idx, (sentence, mask, label) in enumerate(val_dataloader):
                
                sentence, mask, label = sentence.to(device),mask.to(device), label.to(device)
                output = model(input_ids=sentence, attention_mask=mask).logits
                loss = loss_function(output, label)

                for i in np.argmax(output.cpu(), axis=1).tolist():
                    output_list.append(i)
                for i in np.argmax(label.cpu(), axis=1).tolist():
                    label_list.append(i)

                val_losses.append(loss.item())
                tf_logger.add_scalar('validation loss', loss.item(), step)
                step += 1
                progress.set_postfix({
                    'val_loss': np.mean(val_losses),
                    })
                progress.update()
            tf_logger.add_scalar('validation loss', np.mean(val_losses), epoch)
            progress.close()

            #save model (macro-f1 / loss)
            f1 = f1_score(torch.tensor(output_list), torch.tensor(label_list), average='macro')
            if f1 > best_f1:
                no_progress_step = 0 #重置
                torch.save(
                    model.state_dict(),
                    f"./models/{model_type.split('/')[-1]}-f1.pt"    
                )
                print(f">>> f1_score={f1} > {best_f1}, model saved ")
                best_f1 = f1

            else:
                print(f">>> f1_score={f1} < {best_f1}, don't save the model")
                no_progress_step += 1
                print(f">>> The loss accumulate {no_progress_step} times have no progress")
                if no_progress_step == 10:
                    break

            # if np.mean(val_losses) < best_loss:
            #     no_progress_step = 0 #重置

            #     torch.save(
            #         model.state_dict(),
            #         f"./models/{model_type.split('/')[-1]}-loss.pt"    
            #     )
            #     print(f">>> model saved, epoch={epoch}, val_loss={np.mean(val_losses)}")

            #     best_loss = np.mean(val_losses)
            # else:
            #     print(f">>> val_loss={np.mean(val_losses)} > {best_loss}, don't save the model")
            #     no_progress_step += 1
            #     print(f">>> The loss accumulate {no_progress_step} times have no progress")
            #     if no_progress_step == 10:
            #         break