import pandas as pd
from collections import Counter

data = pd.read_csv('./COS_train.csv')
# print(data.head())
# label_count = Counter([label_for_kaggle for label_for_kaggle in data['label_for_kaggle']])
# print(label_count)
print(data['sentence'].values[1])