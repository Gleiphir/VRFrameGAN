import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from mydataloader import FrameDiffDataset
import model
import numpy as np



F = FrameDiffDataset(r"myData/210110-demo-split/L")
loader = DataLoader(F,batch_size=1)

def running_mean(array,window_len):
    return np.convolve(array, np.ones(window_len) / window_len, mode='valid')

def toNpImg(t:torch.Tensor):
    fst,scd = np.split(t.detach().cpu().squeeze().numpy(),2)
    return fst.transpose(1,2,0) + 0.5 ,scd.transpose(1,2,0) +0.5


G = model.Generator().cuda()
G.load_state_dict(torch.load("nets/crop-Jan21-2/G_500",map_location=torch.device('cuda:0')))

import matplotlib.pyplot as plt

lossrecD = np.load("nets/crop-Jan21-2/lossD.npy")
lossrecG = np.load("nets/crop-Jan21-2/lossG.npy")

lossD = running_mean(lossrecD,10)
lossG = running_mean(lossrecG,10)

plt.plot(lossD,label="D loss")
plt.plot(lossG,label="G loss")
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize=18)
plt.title("Loss over time (10-batch running mean)", fontsize=18)
plt.show()

plt.clf()

for x,y in loader:
    x = x.cuda()
    fake1,fake2 = toNpImg(G(x))
    y1,y2 = toNpImg(y)
    print(np.min(y1),np.max(y1))
    print(np.min(fake1), np.max(fake2))
    ax = plt.subplot(2, 2, 1 )
    ax.set_title('Fake 1')
    ax.imshow(fake1)
    ax = plt.subplot(2, 2, 2)
    ax.set_title('Fake 2')
    ax.imshow(fake2)
    ax = plt.subplot(2, 2, 3 )
    ax.set_title('Real 1')
    ax.imshow(y1)
    ax = plt.subplot(2, 2, 4)
    ax.set_title('Real 2')
    ax.imshow(y2)
    plt.show()
    input()