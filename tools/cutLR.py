import cv2
from pathlib import Path
from pathlib import PosixPath
Ifolder = Path(r"D:\Github\VRFrameGAN\myData\210110-demo")
Ofolder = Path(r"D:\Github\VRFrameGAN\myData\210110-demo-split")

def getind(P:Path):
    return int(str(P.name).split('.')[0])

(Ofolder / "L" ).mkdir(parents=True, exist_ok=True)
(Ofolder / "R" ).mkdir(parents=True, exist_ok=True)

for im in Ifolder.glob("*.png"):
    #print(str(im.absolute()))
    img = cv2.imread(str(im.absolute()))
    h, w, c = img.shape
    ind = getind(im)
    imgL = img[:, 0: (w // 2+1)]
    imgR = img[:,  (w // 2+1) : -1]
    cv2.imwrite(str(Ofolder / r"L\{0:04d}.png".format(ind)),imgL)
    cv2.imwrite(str(Ofolder / r"R\{0:04d}.png".format(ind)),imgR)
    print("Proceeded img No. ",ind)
    #cv2.imshow("L", imgL)
    #cv2.imshow("R", imgR)
    #cv2.waitKey(0)