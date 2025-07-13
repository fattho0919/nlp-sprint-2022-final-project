import torch
from torchmetrics.classification import BinaryF1Score
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, BloomForSequenceClassification, GPT2LMHeadModel,  BertTokenizerFast, AutoModel
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig 
import loralib as lora
import pandas as pd
from collections import Counter
from torch.utils.data import random_split
from tqdm import tqdm

class all_dataset(Dataset):
    def __init__(self, file_path, type, tokenizer):
        self.tokenizer = tokenizer
        self.sentence_token = []
        self.type = type
        self.data = pd.read_csv(file_path)
        self.sentence_list = ['<s>' + sentence + '</s>' for sentence in self.data['sentence'].values]
        
        if self.type == 'train':
            self.label_list = [label for label in self.data['label_for_kaggle'].values]
            label_count = Counter(self.label_list)
            print("Total data size is %0d" % len(self.data), f"and {label_count}")

        self.sentence_token = self.tokenizer(self.sentence_list, padding=True, return_tensors='pt')
        # print(self.sentence_token['input_ids'][0])
        # print('\n',self.sentence_token['attention_mask'][0])
        # exit()


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
            return tokens, mask, label
        else:
            return tokens, mask
        

epochs = 200
bs = 2
# model_type = 'ckip-joint/bloom-1b1-zh'
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-pos')
device = torch.device('cuda:0')

train_file_path = './COS_train.csv'
test_file_path = './COS_test.csv'
train_dataset, val_dataset = random_split(all_dataset(train_file_path, 'train', tokenizer), [0.8, 0.2], generator=torch.Generator().manual_seed(42))
test_dataset = all_dataset(test_file_path, 'test', tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True, drop_last=True)

# config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     target_modules=["query_key_value"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="SEQ_CLS"
# )

# model = get_peft_model(model, config)
# model.print_trainable_parameters()

# i = 0
# for param in model.parameters():
#     i += 1
#     if i < 2:
#         param.requires_grad =False
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=20, num_training_steps=epochs * len(train_dataloader)
    )
# save_config(args)
step = 0
no_progress_step = 0
best_loss = float('inf')
best_f1 = 0
tf_logger = SummaryWriter(log_dir=f'log/BLOOM')

for epoch in range(epochs):
        model.train()
        print(f">>> Epochs {epoch + 1}")
        train_losses = []
        progress = tqdm(total=len(train_dataloader), desc='training_progress')

        for batch_idx, (sentence, mask, label) in enumerate(train_dataloader):
            model.zero_grad()
            sentence, mask, label = sentence.to(device),mask.to(device), label.to(device)

            output = model(input_ids=sentence, attention_mask=mask).logits
            loss = criterion(output, label)

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
            metric = BinaryF1Score()
            for batch_idx, (sentence, mask, label) in enumerate(val_dataloader):

                sentence, mask, label = sentence.to(device),mask.to(device), label.to(device)
                output = model(input_ids=sentence, attention_mask=mask).logits
                loss = criterion(output, label)

                for i in np.argmax(output.cpu(), axis=1).tolist():
                    output_list.append(i)
                for i in label.tolist():
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

            f1_score = metric(torch.tensor(output_list), torch.tensor(label_list))
            if f1_score > best_f1:
                no_progress_step = 0 #重置
                model.save_pretrained("./models/BLOOM_LoRA"  )
                # torch.save(
                #     model.state_dict(),
                #     "./models/BLOOM_LoRA.pt"    
                # )
                print(f">>> model saved, epoch={epoch}, f1_score={f1_score}")

                best_f1 = f1_score
            else:
                print(f">>> f1_score={f1_score} < {best_f1}, don't save the model")
                no_progress_step += 1
                print(f">>> The loss accumulate {no_progress_step} times have no progress")
                if no_progress_step == 10:
                    break

            # if np.mean(val_losses) < best_loss:
            #     no_progress_step = 0 #重置
            #     # model.save_pretrained("./models/BLOOM_LoRA"  )
            #     torch.save(
            #         model.state_dict(),
            #         "./models/BLOOM_LoRA.pt"    
            #     )
            #     print(f">>> model saved, epoch={epoch}, val_loss={np.mean(val_losses)}")

            #     best_loss = np.mean(val_losses)
            # else:
            #     print(f">>> val_loss={np.mean(val_losses)} > {best_loss}, don't save the model")
            #     no_progress_step += 1
            #     print(f">>> The loss accumulate {no_progress_step} times have no progress")
            #     if no_progress_step == 10:
            #         break