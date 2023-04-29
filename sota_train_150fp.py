import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np

from time import time
import sys
from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from framework import MyFrame
from loss import dice_bce_loss
from gt_data import ImageFolder
if __name__ =='__main__':
    SHAPE = (1024,1024)
    ROOT = './baidu/train_spadedirty/'
    imagelist = filter(lambda x: x.find('jpg')!=-1, os.listdir(ROOT))
    trainlist=[]
    for image in list(imagelist):
        trainlist.append(image[:-4])
    # print(trainlist)
    NAME = 'gpu0'
    BATCHSIZE_PER_CARD = 4

    solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
    batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
    print(batchsize)
    dataset = ImageFolder(trainlist, ROOT)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4)

    mylog = open('logs/'+NAME+'.log','w+')
    tic = time()
    no_optim = 0
    total_epoch = 150
    train_epoch_best_loss = 100.
    # solver.load('./SOTA/sota.th')
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
        if(epoch<=100):
            print('norm')
            if train_epoch_loss >= train_epoch_best_loss:
                no_optim += 1
            else:
                no_optim = 0
                train_epoch_best_loss = train_epoch_loss
                solver.save('./SOTA/' + NAME + '.th')
            if no_optim > 6:
                print(mylog, 'early stop at %d epoch' % epoch)
                print('early stop at %d epoch' % epoch)
                break
            if no_optim > 3:
                if solver.old_lr < 5e-7:
                    break
                solver.load('./SOTA/' + NAME + '.th')
                solver.update_lr(5.0, factor=True, mylog=mylog)
        else:
            print('poly')
            solver.adjust_learning_rate_poly(epoch,total_epoch,4)
            solver.save('./SOTA/gpu0.th')
        mylog.flush()

    mylog.write('Finish!')
    print( 'Finish!')
    mylog.close()