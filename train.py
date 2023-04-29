import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
from time import time

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
if __name__ =='__main__':
    SHAPE = (1024,1024)
    imgline,maskline=[],[]
    # ROOT = 'dataset/train/'
    with open('./baidu/img_paths_txt/total_img.txt', 'r', encoding='utf8') as f:
        linea = f.readlines() # baidu/河南省/平顶山市/mask/4458.png
    for img in linea:
        imgline.append(img.strip())

    # ROOT=line
    with open('./baidu/img_paths_txt/total_mask.txt', 'r', encoding='utf8') as f:
        lineb = f.readlines()
    for img in lineb:
        maskline.append(img.strip())

    # imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))
    # imagelist=line
    # trainlist=[]
    # for image in list(imagelist):
    #     trainlist.append(image[:-8])
    # print(trainlist)
    trainlist=[]
    # for img in line:
    #     trainlist.append(img.split('/')[-1].strip())
        # trainlist.append(img.split(("/")[-1][:-1]))
    # NAME = 'log01_dink34'
    NAME='total_image1'
    BATCHSIZE_PER_CARD = 4

    solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
    batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

    dataset = ImageFolder(imgline,maskline)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4)

    mylog = open('logs/'+NAME+'.log','w')
    tic = time()
    no_optim = 0
    total_epoch = 100
    train_epoch_best_loss = 100.
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(data_loader_iter)
        mylog.write( '********'+"\n")
        mylog.write( 'epoch:'+str(epoch)+'    time:'+str(time()-tic)+'\n')
        mylog.write( 'train_loss:'+str(train_epoch_loss)+'\n')
        mylog.write('SHAPE:'+str(SHAPE)+'\n')
        print( '********')
        print ('epoch:',epoch,'    time:',int(time()-tic))
        print ('train_loss:',train_epoch_loss)
        print( 'SHAPE:',SHAPE)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('weights/'+NAME+'.th')
        if no_optim > 6:
            print(mylog, 'early stop at %d epoch' % epoch)
            print( 'early stop at %d epoch' % epoch)
            break
        if no_optim > 3:
            if solver.old_lr < 5e-7:
                break
            solver.load('weights/'+NAME+'.th')
            solver.update_lr(5.0, factor = True, mylog = mylog)
        mylog.flush()

    print(mylog, 'Finish!')
    print( 'Finish!')
    mylog.close()