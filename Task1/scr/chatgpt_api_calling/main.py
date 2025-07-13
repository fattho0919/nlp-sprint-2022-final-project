# -*- coding: utf-8 -*-
import pandas as pd
import openai
import json
import numpy as np

test_data = pd.read_csv("./COS_test.csv")
test_data['sentence'] = test_data['sentence'].str.strip('.')
test_data['sentence'] = test_data['sentence'].str.strip()
sentence_list = test_data['sentence'].values
source_id_list = test_data['source_id'].values

output_log = {
    'source_id': [],
    'label_for_kaggle':[]
}

for source_id, sentence in zip(source_id_list, sentence_list):
    prompt = "\n請判斷以下廣告是否違反以下任何一條項目：\n\n1. 有傷風化、違背公共秩序善良風俗；\n2. 名稱、製法、效用或性能虛偽誇大；\n3. 保證其效用或性能；\n4. 涉及疾病治療或預防。\n\n廣告內容如下：\n" + f"「{sentence}」。" +"\n\n請以一個字回答：「是」或「否」。\n\n\n"
    params = {"stop":[" end"]}
    response = openai.Completion.create(model="ada:ft-personal-2023-05-18-16-11-24", prompt=prompt, **params)
    ans = response["choices"][0]["text"].strip()
    print(ans)
    output_log['source_id'].append(source_id)
    if ans == '是':
        output_log['label_for_kaggle'].append(0)
    else:
        output_log['label_for_kaggle'].append(1)

df = pd.DataFrame(output_log)
df.to_csv('./0519_chatgpt_fine_tuned.csv',index=False)
