import numpy as np

folder_real = 'data'
file_name = ['HeadData', 'Lfootdata', 'Rfootdata', 'WaistData']
file_type = '.txt'

height_data = [[], [], [], []]

for i, file in enumerate(file_name):
    f = open(folder_real + '/' + file + file_type, 'r')

    while True:
        line = f.readline()
        if not line: break
        height_data[i].append(float(line))
    f.close()

train_data = []
train_label = []

for i in range(len(height_data[0]) - 10):
    train_data.append(height_data[1][i:i+10]+height_data[2][i:i+10])
    train_label.append([height_data[0][i], height_data[3][i]])

train_data = np.array(train_data)
train_label = np.array(train_label)

np.savez('train', foot_train_data = train_data, foot_train_label = train_label)