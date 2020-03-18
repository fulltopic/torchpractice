import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw(y, color, i):
    # print(y.size(0))
    print(y)
    plt.plot(np.arange(y.size), y, color, linewidth=2.0)
    plt.savefig("./testplot%d.pdf"%i)
    plt.close()

if __name__ == '__main__':
    # data = torch.load('traindata.pt')
    data = torch.load('testreaddata.pt')
    print('data shape')
    print(data.shape)
    input = torch.from_numpy(data)
    print(input.size(0))
    for i in range(input.size(0)):
        example = data[i, :]
        print(example.shape)
        draw(example, 'g', i)