import torch
import os
import pathlib
from pathlib import Path
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as trs
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

import tools.imgMomentum as M

#toTorch = lambda im : im.transpose((2, 0, 1))

ImProcessDeck = trs.Compose([
    trs.RandomCrop((96,108)),
    #trs.Normalize((0.5, 0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5, 0.5) )
])
"""
ImProcessFig = trs.Compose([
    trs.RandomCrop(128,padding=4),

    #trs.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5, 0.5) )
])
"""

class FrameDiffDataset(Dataset):
    """Frame diff Dataset."""

    def __init__(self, root_dir:str):

        """
        Args:
            root_dir (string): Directory with all the images.

        """
        self.imList = list(Path(root_dir).iterdir())

    def __len__(self):
        return len(self.imList) - 3

    def __getitem__(self, idx:int):
        im1 = M.rpicNorm(self.imList[idx]).transpose((2,0,1))
        im2 = M.rpicNorm(self.imList[idx+1]).transpose((2,0,1))
        im3 = M.rpicNorm(self.imList[idx+2]).transpose((2,0,1))
        im4 = M.rpicNorm(self.imList[idx+3]).transpose((2,0,1))
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        noise = np.expand_dims( np.random.normal(0.5,0.5,size=im1[0].shape),axis=0)

        diff = M.momentumMap(im1,im4)
        mergedX = np.concatenate((im1,diff,noise),axis=0)

        stackedY = np.concatenate((im2,im3),axis=0)
        Processed = ImProcessDeck(torch.cat([torch.Tensor(mergedX),torch.Tensor(stackedY)]))
        # Normalized Images.
        return Processed[:5,:,:] , Processed[5:,:,:]


if __name__ == "__main__":
    F = FrameDiffDataset(r"D:\Github\VRFrameGAN\myData\210110-demo-split\L")
    loader = DataLoader(F)
    for (x,y) in loader:
        print(x.size(),y.size())
        input()