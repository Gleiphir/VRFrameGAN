import PIL
from PIL import Image
import numpy as np
import skimage.measure
from scipy import ndimage
import matplotlib.pyplot as plt



non0mean = lambda ary: np.sum(ary) / np.count_nonzero(ary)

rpic = lambda path: np.array(Image.open(path)).astype(np.int16)

rpicNorm = lambda path: np.array(Image.open(path),dtype=np.float) / 255.0   #[0.0 , 1.0]

# (row (height), column (width), color (channel))

def momentum(this:np.ndarray,future:np.ndarray):
    Max = []
    Min = []
    for i in range(3):
        Max.append(np.abs(ndimage.maximum_filter(future[:,:,i],size=2) - ndimage.maximum_filter(this[:,:,i],size=2)))
        Min.append(np.abs(ndimage.minimum_filter(future[:,:,i],size=2) - ndimage.minimum_filter(this[:,:,i],size=2)))
    return np.array(Max).transpose((1,2,0)).astype(np.uint8),np.array(Min).transpose((1,2,0)).astype(np.uint8)


def momentumMap(this:np.ndarray,future:np.ndarray):
    # numpy image: H x W x C
    # torch image: C X H X W <- Use this
    Max = []
    Min = []
    for i in range(3):
        Max.append(np.abs(ndimage.maximum_filter(future[i,:,:],size=2) - ndimage.maximum_filter(this[i,:,:],size=2)))
        Min.append(np.abs(ndimage.minimum_filter(future[i,:,:],size=2) - ndimage.minimum_filter(this[i,:,:],size=2)))
    grey = np.expand_dims( np.sum( np.sum( (np.array(Max),
                  np.array(Min) ),axis=0 ),axis=0),axis=0)
    return ndimage.maximum_filter(np.where(grey > non0mean(grey), 1.0, 0.0),size=2)


if __name__ =="__main__":
    _1 = rpicNorm(r"D:\Github\VRFrameGAN\myData\210110-demo-split\L\0000.png")
    print(np.min(_1),np.max(_1))


    _1 = rpic(r"D:\Github\VRFrameGAN\myData\210110-demo-split\L\0000.png")
    _3 = rpic(r"D:\Github\VRFrameGAN\myData\210110-demo-split\L\0004.png")

    _diffmax,_diffmin = momentum(_1,_3)
    #plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9)
    fig = plt.figure()
    fig.tight_layout()

    ax1 = fig.add_subplot(241)  # left side
    ax1.set_title("Frame 0")
    ax2 = fig.add_subplot(242)  # right side
    ax2.set_title("Frame 4")
    ax3 = fig.add_subplot(243)
    ax3.set_title("max filter diff (5x)")
    ax4 = fig.add_subplot(244)
    ax4.set_title("min filter diff (5x)")
    ax5 = fig.add_subplot(245)
    ax5.set_title("scaled max diff")
    ax6 = fig.add_subplot(246)
    ax6.set_title("scaled min diff")
    ax7 = fig.add_subplot(247)
    ax7.set_title("binary diff")
    ax8 = fig.add_subplot(248)
    ax8.set_title("hotspot area")

    ax1.imshow(_1)


    ax2.imshow(_3)

    ax3.imshow(_diffmax * 5)
    ax4.imshow(_diffmin * 5)

    ax5.imshow(np.where(_diffmax > non0mean(_diffmax),255,0))
    ax6.imshow(np.where(_diffmin > non0mean(_diffmin), 255, 0))

    diff = np.sum(_diffmax + _diffmin,axis = 2)
    diffGrey = np.where(diff > non0mean(diff), 255, 0)
    ax7.imshow(diffGrey)
    diffRobust = ndimage.maximum_filter(diffGrey,size=2)
    ax8.imshow(diffRobust)

    plt.show()