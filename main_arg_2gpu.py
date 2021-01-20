import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from mydataloader import FrameDiffDataset
import model
import numpy as np


scale_factor =4




BATCH_SIZE = 5

GPU_G = 1
GPU_D = 2
GPU_loss = 0

def cudaN(ind:int):
    return torch.device(type='cuda', index=ind)

F = FrameDiffDataset(r"myData/210110-demo-split/L")
loader = DataLoader(F,batch_size=BATCH_SIZE ,shuffle=True)

criterion = nn.BCELoss()

G = model.Generator().cuda(device=cudaN(GPU_G))
D = model.Discriminator().cuda(device=cudaN(GPU_D))

optG = Adam(G.parameters(),lr=0.00002,betas=(0.5,0.999))
optD = Adam(D.parameters(),lr=0.00002,betas=(0.5,0.999))

real_label = 1.0
fake_label = 0.0

lossrecD = []
lossrecG = []

def train(deckX,realY):
    deckX = deckX.cuda(device=cudaN(GPU_G))
    realY = realY.cuda(device=cudaN(GPU_D))
    # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    optD.zero_grad()
    lossD = criterion(D(realY), torch.ones(BATCH_SIZE, 1).cuda(device=cudaN(GPU_D))) \
            + criterion(D(G(deckX).to(device=cudaN(GPU_D))),torch.zeros((BATCH_SIZE, 1)).cuda(device=cudaN(GPU_D)))

    # 0.0-1.0 higher = more real
    lossD.backward()
    optD.step()
    lossrecD.append(lossD.detach().item())
    print("Loss D: ", lossD.detach().item())

    ## (2) Update G network: maximize log(D(G(z)))
    optG.zero_grad()
    lossG = criterion(D(G(deckX).to(device=cudaN(GPU_D))), torch.ones(BATCH_SIZE, 1).cuda(device=cudaN(GPU_D)))  # G wants D to predict 1.0
    lossG.backward()
    optG.step()
    lossrecG.append(lossG.detach().item())
    print("Loss G: ", lossG.detach().item())


for i,(deck, real) in enumerate(loader):
    train(deck,real)
    print("Epoch {}".format(i))
    if i % 20 ==0:
        torch.save(G.state_dict(),"output/G_{}".format(i))

    if i >= 10000:
        break

np.save("output/lossD.npy",lossrecD)
np.save("output/lossG.npy",lossrecG)
print("Done")
"""
import matplotlib.pyplot as plt
plt.plot(lossrecD)
plt.plot(lossrecG)
plt.show()
"""
