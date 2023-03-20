import numpy as np
import torch 

def get_data():
    with open('processed_data/s7.csv') as f:
        s7=f.readlines()
    with open('processed_data/modbus.csv') as f:
        modbus=f.readlines()
    with open('processed_data/cip.csv') as f:
        cip=f.readlines()

    return [s7, modbus, cip]

def subprocess(data):
    data = data[1:]
    data = [data[i][:-2].split(',')for i in range(len(data))]
    for i in range(len(data)):
        if '' in data[i]:
            data[i].remove('')
        data[i] = list(map(int,data[i])) 
    max_len = max([len(data[i]) for i in range(len(data))])
    data = [data[i] + [-1]*(max_len-len(data[i])) for i in range(len(data))]
    return np.array(data)

def preprocess(data_list):
    total = []
    for data in data_list:
        total.append(subprocess(data))
    max_len = max([data.shape[1] for data in total])
    for i in range(len(total)):
        data = total[i]
        if data.shape[1]<max_len:
            pad = max_len - data.shape[1]
            data = np.concatenate((data, np.ones((len(data),pad))*(-1)),axis=1)
        data = np.concatenate((data, np.ones((len(data),1))*i), axis=1)
        total[i] = data
    data = np.concatenate(total)
    return data

def split_dataset(data):
    sample_num = len(data)
    data = data[np.random.permutation(np.arange(sample_num))]
    train_data = data[:int(0.7*sample_num)]
    eval_data = data[int(0.7*sample_num):int(0.8*sample_num)]
    test_data = data[int(0.8*sample_num):]
    
    return {'train': train_data,
            'eval': eval_data,
            'test': test_data
            }