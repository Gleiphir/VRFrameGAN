import cv2
vidcap = cv2.VideoCapture(r'C:\Users\zhang\Videos\Steamvr\Steamvr 2021.01.10 - 21.22.39.01.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("%d.png" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: {},{}'.format(success,count))
  count += 1
