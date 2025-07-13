import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BloomForSequenceClassification, BitsAndBytesConfig
from torch.utils.tensorboard import SummaryWriter
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig 
import loralib as lora
import pandas as pd
from collections import Counter
from torch.utils.data import random_split
from tqdm import tqdm
import csv
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
        source_id = [str(source_id) for source_id in self.data['source_id'].values][item]

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
            return tokens, mask, source_id
        
#model & tokenizer setting
epochs = 200
bs = 1
model_type = 'ckip-joint/bloom-3b-zh' #'ckip-joint/bloom-1b1-zh' or 'ckip-joint/bloom-3b-zh'
bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
tokenizer = AutoTokenizer.from_pretrained(model_type)
model = BloomForSequenceClassification.from_pretrained(model_type, num_labels=2, quantization_config=bnb_config, device_map={"":0})
device = torch.device('cuda:0')

#dataset setting
test_file_path = './COS_test.csv'
test_dataset = all_dataset(test_file_path, 'test', tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=bs)

# Load LoRA model / Fine-tune model
# model = PeftModel.from_pretrained(model, f"./models/BLOOM_LoRA")
model.load_state_dict(torch.load('./models/bloom-3b-zh-f1.pt'))

output_list = []
label_list = []
result = []
with torch.no_grad():
    for batch_idx, (sentence, mask, source_id) in enumerate(test_dataloader):

        sentence, mask = sentence.to(device),mask.to(device)
        output = model(input_ids=sentence, attention_mask=mask).logits
        output_label = output.argmax().item()
        result.append([source_id[0], output_label])
print(result)

with open('0603_v4.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(['source_id', 'label_for_kaggle'])
    write.writerows(result)