import numpy as np
import matplotlib.pyplot as plt
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

def analize():
    for file in file_list:
        if int(file.split("_")[1]) > 3 :
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
        xlabel = []
        data = []
        for target in ['nagao', 'ours', 'seo']:
            current = 'real'
            xlabel.append(current + " vs " + target)
            data.append(score_data[target][current] / (score_data[current][target] + score_data[target][current])* 100)
            print(current, 'vs', target, ': ', score_data[target][current] / (score_data[current][target] + score_data[target][current]) * 100)

        for target in ["real",'nagao', 'seo']:
            current = 'ours'
            xlabel.append( current + " vs " + target)
            data.append(score_data[target][current] / (score_data[current][target] + score_data[target][current])* 100)
            print(current, 'vs', target, ': ', score_data[target][current] / (score_data[current][target] + score_data[target][current]) * 100)

        print('---------------\n')
        make_graph(data,xlabel,"percent",tag)
def make_graph(data,xAxisLabel,yAxisName,title,barWidth = 0.5):
    x = np.arange(len(xAxisLabel))
    barWidth = len(xAxisLabel)/10
    plt.xticks(x, xAxisLabel)
    plt.title(title)
    plt.xticks(fontsize=13)
    plt.ylim(0, 110)
    plt.yticks(range(0, 101, 20), fontsize=13)

    plt.ylabel(yAxisName, fontsize=12)
    plt.hlines(50, -barWidth * 1.5, len(xAxisLabel) - (1 - barWidth * 1.5), colors="Red",alpha=0.7, linestyles="--")
    # for i in x:
    #     plt.errorbar(i, data[i], yerr= random.randint(1,3) + random.random() , color="black", elinewidth=0.7, capsize=2)

    plt.xlim(-barWidth * 1.5, len(xAxisLabel) - (1 - barWidth * 1.5))
    plt.grid(True, axis="y", alpha=0.5)
    plt.bar(x, data, width=barWidth,color = "#3C93C2",edgecolor = "C0",alpha=0.8,zorder = 1)

    plt.show()

analize()