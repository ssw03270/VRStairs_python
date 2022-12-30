import numpy as np
import pandas as pd

names = ['박승준']
files = ['chose_1_0.csv', 'chose_1_1.csv', 'chose_1_2.csv', 'chose_1_3.csv']
methods = {'nagao': 0, 'ours': 1, 'seo': 2, 'real': 3}
col_name = ['nagao', 'ours', 'seo', 'real']
row_name = ['nagao', 'ours', 'seo', 'real']

for name in names:
    scores = np.zeros((4, 4))

    for file in files:
        score = np.zeros((4, 4))
        path = './experiment2/' + name + '/' + file
        data = pd.read_csv(path)

        for index, row in data.iterrows():
            if row['chose'] == 'First':
                score[methods[row['first']]][methods[row['second']]] += 1
            elif row['chose'] == 'Second':
                score[methods[row['second']]][methods[row['first']]] += 1

        score_data = pd.DataFrame(score, columns=col_name)
        score_data.index = row_name
        print(score_data)
        scores = scores + score

    scores_data = pd.DataFrame(scores, columns=col_name)
    scores_data.index = row_name
    print(scores_data)