# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:00:34 2019

@author: yuxuan
"""

import torch
import numpy as np
import os
import json
from VSE_CAP import VSE_CAP
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import heapq
import math
from test_data import COCO_Dataset
from torchvision import transforms as T


os.environ["CUDA_VISIBLE_DEVICES"]='1'
torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataf = 'D:/python_project/caption/coco_io.json'
img_dir = 'D:/python_project/data/caption_data/'
vocab_f = 'D:/python_project/caption/coco_vocab.json'


with open(dataf) as f:
    data=json.load(f)
test_set=data['test']

with open(vocab_f) as f:
    vocab=json.load(f)
    vocab=vocab['itow']

transform=T.Compose([
                T.Resize(256), 
                T.RandomCrop(224), 
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])

test_dataset = COCO_Dataset(img_dir=img_dir, coco_io=test_set, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
print(len(test_set))

batch_size=1
max_len=18
model=VSE_CAP(feat_size=300, hidden_size=1024, vocab_size=9488,
              sptial_size=196, channel_size=2048, out_channel=1024)

if torch.cuda.is_available():
    model = model.cuda()
save_f = 's/29_16000.npy'
trained_dict = np.load(save_f, allow_pickle=True).item()
model.load_state_dict(trained_dict)
model.eval()

def greedy_search(word_list, h_, c_):
    greedy_seq = []
    ox = torch.LongTensor([0]).cuda()
    h = h_
    c = c_
    for T in range(max_len-1):
        word_emb = model.CBAM_SCA.w2vCap(ox)
        h,c = model.CBAM_SCA.LanguageLSTM(word_emb,(h,c))
        o = model.CBAM_SCA.classfier(h)
        _, ox = torch.topk(o,1)
        ox = ox[0]
        if ox == torch.LongTensor([0]).cuda():
            break
        greedy_seq.append(word_list[str(int(ox))])
    return ' '.join(greedy_seq)

class caption_meta(object):
    def __init__(self,sentence,state,logprob,score):
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score

def takeScore(elem):
    return elem.score

def beam_search(beam_size, word_list, h_, c_):
    ox = torch.LongTensor([0]).cuda()
    init_beam = caption_meta(sentence=[ox], state=(h_,c_), logprob=0.0, score=0.0)
    in_beam = [init_beam]
    for T in range(max_len-1):
        beam_res = []
        for beam in in_beam:
            if beam.sentence[-1].cpu().data.numpy()[0] == 0 and len(beam.sentence) != 1:
                beam_res.append(beam)
            else:
                word_emb = model.CBAM_SCA.w2vCap(beam.sentence[-1])
                h,c = model.CBAM_SCA.LanguageLSTM(word_emb,beam.state)
                o = model.CBAM_SCA.classfier(h)
                o = F.softmax(o,dim=1)
                p,ox = torch.topk(o,beam_size)
                ox = ox[0]
                p = p[0]
                for i in range(beam_size):
                    logprob = math.log(p[i])
                    score = beam.score+math.log(p[i])
                    sentence = beam.sentence[:]
                    sentence.append(ox[i].view(1))
                    new_beam = caption_meta(sentence=sentence,state=(h,c),logprob=logprob,score=score)
                    beam_res.append(new_beam)
        beam_res.sort(key=takeScore, reverse=True)
        in_beam = beam_res[:beam_size]
    all_cap = []
    for beam in in_beam:
        beam_cap = []
        for word in beam.sentence[1:]:
            if word == torch.LongTensor([0]).cuda():
                break
            beam_cap.append(vocab[str(int(word))])
        all_cap.append(' '.join(beam_cap))
    return all_cap

aa = []
for j, data in enumerate(test_loader):
    I = data['image']
    N = data['img_name']
    N = N[0]
    print(N)
    I = Variable(I).to(device)
    init_fs = model.CNN(I)
    init_fs = model.CBAM_SCA.avgpool(init_fs)
    init_fs = init_fs.view(init_fs.shape[0], -1)
    h, c = model.CBAM_SCA.init_hidden_state(init_fs)
    #gen_seq=greedy_search(vocab,h,c)
    gen_seq = beam_search(3, vocab, h, c)[0]
    print(gen_seq)
    captiondict = {"image_id": int(N.split('/')[-1][-10:-4]), "caption": gen_seq}
    aa.append(captiondict)

with open('CAP_BASE_beam.json', 'w') as json_file:
    json.dump(aa,json_file)
    
    

        

    

