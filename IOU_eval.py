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


def makdir(path):
    str = '.'
    lable = path.split('/')[-1].strip()
    # print(lable)
    for i in range(1,len(path.split('/')) - 2):
        str = str + '/' + path.split('/')[i]
    return str,lable
with open('./baidu/img_paths_txt/total_mask.txt', 'r', encoding='utf8') as f:
    gtlist = f.readlines()
maskpath1='./submits/total_image1'
maskpath2='./submits/total_image2'
list2=os.listdir(maskpath1)
# print(makdir('baidu/河南省/郑州市/二七/mask/1119.png'))
# print(os.listdir('./submits/new_mask'))
# print('./'+gtlist[1])
# ig=cv2.imdecode(np.fromfile('./'+gtlist[0].strip()),cv2.IMREAD_GRAYSCALE)
# print(ig)
gtpath='./baidu/total_image_groundtruth/'
testpath='./baidu/groundtruth/test/'
trainpath='./baidu/groundtruth/train/'
os.mkdir('./baidu/groundtruth/')
os.mkdir(testpath)
os.mkdir(trainpath)
testcnt=0
traincnt=0
for mask in gtlist:
    if makdir(mask)[1].strip() in list2:
        # print('./'+mask.strip())
        # print(maskpath+'/'+makdir(mask)[1])
        result1=calIOU('./'+mask.strip(),maskpath1+'/'+makdir(mask)[1])
        result2=calIOU('./'+mask.strip(),maskpath2+'/'+makdir(mask)[1])
        IOU[makdir(mask)[1]]=result2
        # print(f'rankIOU[{makdir(mask)[1]}]:',result)
        if result1>=0.3 or result2>=0.3:
            if result1>=0.6 and result2>=0.6:
                testcnt+=1
                print('test')
                print(f'rankIOU[{makdir(mask)[1]}]:', result1,result2)
                shutil.copy('./'+mask.strip(),testpath+makdir(mask)[1].strip())
                shutil.copy(makdir('./'+mask.strip())[0]+'/影像/'+makdir(mask)[1].strip()[:-3]+'jpg',testpath+makdir(mask)[1].strip()[:-3]+'jpg')
                print('./'+mask.strip())
                print(makdir('./'+mask.strip())[0]+'/影像/'+makdir(mask)[1].strip()[:-3]+'jpg')
            else:
                traincnt+=1
                print('train')
                print(f'rankIOU[{makdir(mask)[1]}]:', result1,result2)
                shutil.copy('./'+mask.strip(),trainpath+makdir(mask)[1].strip())
                shutil.copy(makdir('./'+mask.strip())[0]+'/影像/'+makdir(mask)[1].strip()[:-3]+'jpg',trainpath+makdir(mask)[1].strip()[:-3]+'jpg')
                print('./'+mask.strip())
                print(makdir('./'+mask.strip())[0]+'/影像/'+makdir(mask)[1].strip()[:-3]+'jpg')
print('test:',testcnt)
print('train',traincnt)
# rankIOU=sorted(IOU.items(),key=lambda x:x[1],reverse=True)
# with open('total_image1_IOU.txt', 'w')as f:
#     for i in range(len(rankIOU)):
#         f.write(rankIOU[i][0].ljust(20,' ')+str(rankIOU[i][1])+'\n')
# tenper=0
# twenper=0
# thirper=0
# fourper=0
# fifper=0
# sixper=0
# sevper=0
# eigper=0
# ninper=0
# hunper=0
# with open('total_image1_STA.txt', 'w')as f:
#     for i in range(len(rankIOU)):
#         score=rankIOU[i][1]
#         if(score<=0.1):
#             tenper+=1
#         elif(0.1<score<=0.2):
#             twenper+=1
#         elif(0.2<score<=0.3):
#             thirper+=1
#         elif(0.3<score<=0.4):
#             fourper+=1
#         elif(0.4<score<=0.5):
#             fifper+=1
#         elif(0.5<score<=0.6):
#             sixper+=1
#         elif(0.6<score<=0.7):
#             sevper+=1
#         elif(0.7<score<=0.8):
#             eigper+=1
#         elif(0.8<score<=0.9):
#             ninper+=1
#         elif(0.9<score<=1.0):
#             hunper+=1
#     f.write("[0,0.1]".ljust(20, ' ') + str(tenper)+'\n')
#     f.write("[0.1,0.2]".ljust(20, ' ') +str(twenper)+'\n')
#     f.write("[0.2,0.3]".ljust(20, ' ') +str (thirper)+'\n')
#     f.write("[0.3,0.4]".ljust(20, ' ') + str(fourper)+'\n')
#     f.write("[0.4,0.5]".ljust(20, ' ') + str(fifper)+'\n')
#     f.write("[0.5,0.6]".ljust(20, ' ') + str(sixper)+'\n')
#     f.write("[0.6,0.7]".ljust(20, ' ') + str(sevper)+'\n')
#     f.write("[0.7,0.8]".ljust(20, ' ') + str(eigper)+'\n')
#     f.write("[0.8,0.9]".ljust(20, ' ') +str (ninper)+'\n')
#     f.write("[0.9,1.0]".ljust(20, ' ') + str(hunper)+'\n')
# print(rankIOU)
#gt数据集写入
# for i in range(len(rankIOU)):
#     if(rankIOU[i][1]>=30.0):
