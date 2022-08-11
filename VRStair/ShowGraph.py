import matplotlib.pyplot as plt
import numpy as np
import random

def ShowAll():
    folder_real = 'foot_dataset/real'
    folder_virtual = 'foot_dataset/virtual'
    folder_virtual_by_model = 'foot_dataset/virtual_by_model'
    file_name = ['HeadData', 'Lfootdata', 'Rfootdata', 'WaistData']
    file_type = '.txt'

    position_data_real = [[], [], [], []]
    position_data_virtual = [[], [], [], []]
    position_data_virtual_by_model = [[], [], [], []]

    for i, file in enumerate(file_name):
        f = open(folder_real + '/' + file + file_type, 'r')

        while True:
            line = f.readline()
            if not line: break
            line = line.replace("(", "").replace(")", "").replace(",", "")
            line = line.split()
            position_data_real[i].append(float(line[1]))

        f.close()

    for i, file in enumerate(file_name):
        f = open(folder_virtual + '/' + file + file_type, 'r')

        while True:
            line = f.readline()
            if not line: break
            position_data_virtual[i].append(float(line))

        f.close()

    for i, file in enumerate(file_name):
        f = open(folder_virtual_by_model + '/' + file + file_type, 'r')

        while True:
            line = f.readline()
            if not line: break
            position_data_virtual_by_model[i].append(float(line))

        f.close()

    x_real = np.arange(len(position_data_real[0]))
    x_virtual = np.arange(len(position_data_virtual[0]))
    x_virtual_by_model = np.arange(len(position_data_virtual_by_model[0]))

    position_data_real = np.array(position_data_real)
    position_data_virtual = np.array(position_data_virtual)
    position_data_virtual_by_model = np.array(position_data_virtual_by_model)

    for i in range(4):
        position_data_real[i] -= position_data_real[i][0]
        position_data_virtual[i] -= position_data_virtual[i][0]
        position_data_virtual_by_model[i] -= position_data_virtual_by_model[i][0]

    real_index_start = 35
    virtual_index_start = 10
    virtual_by_model_index_start = 5

    x_real -= real_index_start
    x_virtual -= virtual_index_start
    x_virtual_by_model -= virtual_by_model_index_start

    index = 2

    plt.plot(x_real[real_index_start:], position_data_real[index][real_index_start:], x_virtual[virtual_index_start:], position_data_virtual[index][virtual_index_start:],
             x_virtual_by_model[virtual_by_model_index_start:], position_data_virtual_by_model[index][virtual_by_model_index_start:])
    plt.legend(['real', 'virtual', 'virtual_by_model'])
    plt.title(file_name[index])
    plt.show()

def ShowReal():
    folder_real = 'foot_dataset/long_real'
    file_name = ['HeadData', 'Lfootdata', 'Rfootdata', 'WaistData']
    file_type = '.txt'

    position_data_real = [[], [], [], []]

    for i, file in enumerate(file_name):
        f = open(folder_real + '/' + file + file_type, 'r')

        while True:
            line = f.readline()
            if not line: break
            position_data_real[i].append(float(line))

        f.close()

    x_real = np.arange(len(position_data_real[0]))

    position_data_real = np.array(position_data_real)

    index_length = 500
    index = random.randrange(len(position_data_real[0]) - index_length)
    real_index_start = index
    real_index_end = index + index_length

    x_real -= real_index_start

    plt.plot(x_real[real_index_start:real_index_end], position_data_real[0][real_index_start:real_index_end],
             x_real[real_index_start:real_index_end], position_data_real[1][real_index_start:real_index_end],
             x_real[real_index_start:real_index_end], position_data_real[2][real_index_start:real_index_end],
             x_real[real_index_start:real_index_end], position_data_real[3][real_index_start:real_index_end])
    plt.legend(['HeadData', 'Lfootdata', 'Rfootdata', 'WaistData'])
    plt.title('real')
    plt.grid(True)
    plt.xticks(np.arange(0, index_length, 10))
    plt.show()

def ShowVirtual():
    folder_real = 'foot_dataset/virtual'
    file_name = ['HeadData', 'Lfootdata', 'Rfootdata', 'WaistData']
    file_type = '.txt'

    position_data_real = [[], [], [], []]

    for i, file in enumerate(file_name):
        f = open(folder_real + '/' + file + file_type, 'r')

        while True:
            line = f.readline()
            if not line: break
            position_data_real[i].append(float(line))

        f.close()

    x_real = np.arange(len(position_data_real[0]))

    position_data_real = np.array(position_data_real)

    index_length = 500
    index = random.randrange(len(position_data_real[0]) - index_length)
    real_index_start = index
    real_index_end = index + index_length

    x_real -= real_index_start

    plt.plot(x_real[real_index_start:real_index_end], position_data_real[0][real_index_start:real_index_end],
             x_real[real_index_start:real_index_end], position_data_real[1][real_index_start:real_index_end],
             x_real[real_index_start:real_index_end], position_data_real[2][real_index_start:real_index_end],
             x_real[real_index_start:real_index_end], position_data_real[3][real_index_start:real_index_end])
    plt.legend(['HeadData', 'Lfootdata', 'Rfootdata', 'WaistData'])
    plt.title('real')
    plt.grid(True)
    plt.xticks(np.arange(0, index_length, 10))
    plt.show()

if __name__ == "__main__":
    ShowReal()