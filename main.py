import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from mydataloader import FrameDiffDataset
import model_light



BATCH_SIZE = 5

F = FrameDiffDataset(r"myData\210110-demo-split\L")
loader = DataLoader(F,batch_size=BATCH_SIZE ,shuffle=True)

criterion = nn.BCELoss()

G = model_light.Generator().cuda()
D = model_light.Discriminator().cuda()

optG = Adam(G.parameters(),lr=0.00002,betas=(0.5,0.999))
optD = Adam(D.parameters(),lr=0.00002,betas=(0.5,0.999))

real_label = 1.0
fake_label = 0.0

lossrecD = []
lossrecG = []

for i,(deck, real) in enumerate(loader):
    deck = deck.cuda()
    real = real.cuda()
    #Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    optD.zero_grad()

    lossDr =  criterion(D(real),torch.ones(BATCH_SIZE ,1).cuda())

    lossDr.backward()

    lossDf = criterion(D(G(deck)), torch.zeros((BATCH_SIZE , 1)).cuda())

    lossDf.backward()
    optD.step()
    # 0.0-1.0 higher = more real
    lossD  = lossDf + lossDr
    lossrecD.append(lossD.item())
    print("Loss D: ",lossD.item())


    ## (2) Update G network: maximize log(D(G(z)))
    optG.zero_grad()
    lossG = criterion(D(G(deck)),torch.ones(BATCH_SIZE ,1).cuda()) #G wants D to predict 1.0
    lossrecG.append(lossG.item())
    print("Loss G: ", lossG.item())
    lossG.backward()
    optG.step()

    print("Epoch {}".format(i))
    if i >= 300:
        break

import matplotlib.pyplot as plt
plt.plot(lossrecD)
plt.plot(lossrecG)
plt.show()