import torch
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
import pickle
from matplotlib import pyplot as plt
import os
import shutil
# img2=cv2.imread('./baidu/河南省/郑州市/二七/mask/1119.png')
# cv2.imshow('test',img[5:26][:])
# cnt=0
# for i in range(1024):
#     for j in range(1024):
#         if img[i][j]==255:
#             cnt+=1
# print(cnt)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

IOU={}
def calIOU(path1,path2):
    img1=cv2.imdecode(np.fromfile(path1,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np.fromfile(path2, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    intersaction=0
    union=0
    img1[img1<=128] = 0
    img1[img1>128] = 1

    img2[img2<=128] = 0
    img2[img2>128] = 1

    intersaction = np.logical_and(img1, img2).sum()
    union=np.logical_or(img1, img2).sum()

    return intersaction/union
maskpath='./submits/train_spade_dirty/'
#maskpath='./submits/traindataset_test_image/'
# maskpath='./submits/dirty_mask/'
list2=os.listdir(maskpath)
ROOT='./baidu/groundtruth/test/'
imagelist = filter(lambda x: x.find('png') != -1, os.listdir(ROOT))
total=0
cnt=0
for mask in imagelist:
    if mask.strip() in list2:
        cnt+=1
        result=calIOU(maskpath+mask.strip(),ROOT+mask)
        IOU[mask]=result
        total+=result
        print(f'IOU[{mask}]=',result)
        # print(f'rankIOU[{makdir(mask)[1]}]:',result)
rankIOU=sorted(IOU.items(),key=lambda x:x[1],reverse=True)
with open('train_spade_dirty_IOU.txt','w')as f:
    for i in range(len(rankIOU)):
        f.write(rankIOU[i][0].ljust(20,' ')+str(rankIOU[i][1])+'\n')
tenper=0
twenper=0
thirper=0
fourper=0
fifper=0
sixper=0
sevper=0
eigper=0
ninper=0
hunper=0
with open('train_spade_dirty_STA.txt','w')as f:
    for i in range(len(rankIOU)):
        score=rankIOU[i][1]
        if(score<=0.1):
            tenper+=1
        elif(0.1<score<=0.2):
            twenper+=1
        elif(0.2<score<=0.3):
            thirper+=1
        elif(0.3<score<=0.4):
            fourper+=1
        elif(0.4<score<=0.5):
            fifper+=1
        elif(0.5<score<=0.6):
            sixper+=1
        elif(0.6<score<=0.7):
            sevper+=1
        elif(0.7<score<=0.8):
            eigper+=1
        elif(0.8<score<=0.9):
            ninper+=1
        elif(0.9<score<=1.0):
            hunper+=1
    f.write("[0,0.1]".ljust(20, ' ') + str(tenper)+'\n')
    f.write("[0.1,0.2]".ljust(20, ' ') +str(twenper)+'\n')
    f.write("[0.2,0.3]".ljust(20, ' ') +str (thirper)+'\n')
    f.write("[0.3,0.4]".ljust(20, ' ') + str(fourper)+'\n')
    f.write("[0.4,0.5]".ljust(20, ' ') + str(fifper)+'\n')
    f.write("[0.5,0.6]".ljust(20, ' ') + str(sixper)+'\n')
    f.write("[0.6,0.7]".ljust(20, ' ') + str(sevper)+'\n')
    f.write("[0.7,0.8]".ljust(20, ' ') + str(eigper)+'\n')
    f.write("[0.8,0.9]".ljust(20, ' ') +str (ninper)+'\n')
    f.write("[0.9,1.0]".ljust(20, ' ') + str(hunper)+'\n')
print(rankIOU)
print('mIOU=',total/cnt)
#gt数据集写入
# for i in range(len(rankIOU)):
#     if(rankIOU[i][1]>=30.0):
