# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:05:07 2019

@author: yuxuan

Jehonathan
"""

import json
import random
import numpy as np
import torch
from torch import nn, optim
from VSE_CAP import VSE_CAP
from torch.autograd import Variable
import os
from tensorboardX import SummaryWriter
from torchvision import transforms as T
from torch.nn.utils.rnn import pack_padded_sequence
from Loss import matching_loss, ContrastiveLoss
from data import COCO_Dataset
import h5py

os.environ["CUDA_VISIBLE_DEVICES"]='0'
torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataf='coco_io.json'
img_dir='D:/python_project/data/caption_data/'
save_f='s/'
log_f='log/'

#调整学习率
def Adjust_learning_rate(optimizer, decay_rate, freeze=None):
    for i,param_group in enumerate(optimizer.param_groups):
        if i not in freeze:
            param_group['lr'] = param_group['lr'] * decay_rate

#save parms
def Save(save_dir,save_model,epoch,step):
    model_dict=save_model.state_dict()
    data = {k: v for k, v in model_dict.items()}
    save_path = os.path.join(save_dir, str(epoch)+'_'+str(step)+'.npy')
    print((" Saving the model to %s..." % (save_path)))
    np.save(save_path, data)
    print("Model saved.")
    rfile=os.path.join(save_dir, str(epoch-1)+'_'+str(step)+".npy")
    if os.path.exists(rfile):
        os.remove(rfile)

def main():
    #label coco_io.json
    with open(dataf) as f:
        data = json.load(f)
    train_set = data['train']

    #batch_size
    batch_size = 64
    num_workers = 1
    train_step_per_epoch = len(train_set)//batch_size

    #图像增广，预处理
    transform=T.Compose([
                    T.Resize(256), 
                    T.RandomCrop(224), 
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                    ])

    #data_set,loader
    train_dataset = COCO_Dataset(img_dir=img_dir,coco_io=train_set, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)

    epoch = 30
    running_loss = 0.0
    #model
    model=VSE_CAP(feat_size=300, hidden_size=1024, vocab_size=9488, sptial_size=196,
                  channel_size=2048,out_channel=1024)
    if torch.cuda.is_available():
        model=model.cuda()
    model.train()
    '''
    ###load_pretrain###
    model_dict=model.state_dict()
    pretrained_dict=np.load('pretrained.npy',allow_pickle=True).item()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    ###################
    '''
    param_num = 0
    for param in model.parameters():
        param_num = param_num + int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    #损失函数
    criterion_XE = nn.CrossEntropyLoss()

    optimizer = optim.Adam([{'params':model.CNN.parameters(),'lr':1e-5},
                            {'params':model.CBAM_SCA.parameters(),'lr':1e-3}])
    #CNN为预训练模型
    for param in model.CNN.parameters():
        param.requires_grad = False
    writer = SummaryWriter(log_f)

    #第i个epoch, 每个batch第j个data
    for i in range(epoch):
        if i != 0 and i % 5 == 0:
            Adjust_learning_rate(optimizer, 0.8, [0])
        if i != 0 and i % 15 == 0:
            for param in model.CNN.parameters():
                param.requires_grad = True
        for j, data in enumerate(train_loader):
            global_step = i * train_step_per_epoch + j
            
            I = data['image']
            L = data['label']
            M = data['mask']
            I = Variable(I).to(device)
            L = Variable(L).to(device)
            M = Variable(M).to(device)
            _, sort_ind = M.sum(dim=1).sort(dim=0, descending=True)
            I = I[sort_ind]
            L = L[sort_ind]
            M = M[sort_ind]
            total_loss = torch.FloatTensor([0.]).to(device)

            predictions = model(I, L, M)
            predictions = pack_padded_sequence(predictions, M.sum(dim=1)-1, batch_first=True)
            capgt = pack_padded_sequence(L[:,1:], M.sum(dim=1)-1, batch_first=True)
            loss_XE = criterion_XE(predictions.data, capgt.data)
            
            '''
            batch_name=data['gts_index']
            batch_name=[batch_name[xx] for xx in sort_ind]
            #print(batch_name)
            gts=[]
            for name in batch_name:
                gts.append(ref[name])
            
            scst_out=scst(I,gts)
            loss = scst_out['loss'].mean()
            '''

            optimizer.zero_grad()
            total_loss = loss_XE
            running_loss += total_loss.data
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 2.0)
            optimizer.step()
            
            writer.add_scalar('batch_loss', total_loss.data, global_step=global_step)
            
            print('Gstep: %6d; epoch: %d; [%5d] loss: %.3f ; learning_rate: %f, %f' 
                      % (global_step, i, j, total_loss.data, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
            

            if j % 4000 == 0 and j != 0:
                Save(save_f,model,i,j)
                writer.add_scalar('loss',running_loss/4000, global_step=global_step)
                running_loss=0.0
            #break
        #break
    writer.close()


if __name__ == '__main__':
    main()
    print('Finish training')