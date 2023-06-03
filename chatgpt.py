import pandas as pd
import openai
import json
import numpy as np

np.random.seed(42)
data = pd.read_csv("./COS_train.csv")
data = data[['sentence','label_for_kaggle']]
data['sentence'] = data['sentence'].str.strip('.')
data['sentence'] = data['sentence'].str.strip()
data['sentence'] = "問題：\n請判斷以下廣告是否違反以下任何一條項目：\n\n1. 有傷風化、違背公共秩序善良風俗；\n2. 名稱、製法、效用或性能虛偽誇大；\n3. 保證其效用或性能；\n4. 涉及疾病治療或預防。\n\n廣告內容如下：\n「" + data['sentence'] +"」。\n\n請以一個字回答：「是」或「否」。\n\n\n回答："
data.loc[data['label_for_kaggle']==0, ['label_for_kaggle']] = " 是 END"
data.loc[data['label_for_kaggle']==1, ['label_for_kaggle']] = " 否 END"

print(data.head().sentence[0])