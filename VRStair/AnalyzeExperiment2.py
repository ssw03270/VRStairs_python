import numpy as np
import pandas as pd
import os

names = ['박승준']
files = ['chose_1_0.csv', 'chose_1_1.csv', 'chose_1_2.csv', 'chose_1_3.csv']
methods = {'nagao': 0, 'ours': 1, 'seo': 2, 'real': 3}
col_name = ['nagao', 'ours', 'seo', 'real']
row_name = ['nagao', 'ours', 'seo', 'real']

scores = {'stair1_75': np.zeros((4, 4)), 'stair1_100': np.zeros((4, 4)), 'stair2_75': np.zeros((4, 4)), 'stair2_100': np.zeros((4, 4))}
# for file in files:
#     for name in names:
file_list = os.listdir('./experiment2/')
for file in file_list:
    path = './experiment2/' + file
    data = pd.read_csv(path)

    for index, row in data.iterrows():
        if row['chose'] == 'First':
            scores[row['current block']][methods[row['first']]][methods[row['second']]] += 1
        elif row['chose'] == 'Second':
            scores[row['current block']][methods[row['second']]][methods[row['first']]] += 1

for tag in scores:
    print(tag)
    score = scores[tag]
    score_data = pd.DataFrame(score, columns=col_name)
    score_data.index = row_name
    print("전체 스코어 (표의 좌측이 이긴 쪽, 상단이 진 쪽)")
    print(score_data)

    print("\na vs b에서 a가 이긴 비율")
    for target in ['nagao', 'ours', 'seo']:
        current = 'real'
        print(current, 'vs', target, ': ', score_data[target][current] / (score_data[current][target] + score_data[target][current]) * 100)

    for target in ['nagao', 'seo']:
        current = 'ours'
        print(current, 'vs', target, ': ', score_data[target][current] / (score_data[current][target] + score_data[target][current]) * 100)

    print('---------------\n')