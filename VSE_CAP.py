# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:43:28 2019

@author: yuxuan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math
from MyResnet import resnet101


device=torch.device("cuda")

def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X
    

class CBAM_SCA(nn.Module):
    def __init__(self, feat_size, hidden_size, vocab_size, sptial_size, channel_size, out_channel):
        super(CBAM_SCA,self).__init__()
        
        self.minval = -0.1
        self.maxval = 0.1
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.batch_size = None
        self.sptial_size = sptial_size
        self.channel_size = channel_size
        self.out_channel = out_channel
        self.vocab_size = vocab_size
        self.wh = int(math.sqrt(sptial_size))
        self.maxlen = None

        self.Dropout05 = nn.Dropout(p=0.5)
        self.Dropout025 = nn.Dropout(p=0.25)
        self.f2h = nn.Linear(self.channel_size, self.hidden_size)
        self.f2c = nn.Linear(self.channel_size, self.hidden_size)
        self.LanguageLSTM = nn.LSTMCell(self.feat_size, self.hidden_size, bias=True)
        self.classfier = nn.Linear(self.hidden_size, self.vocab_size)
        self.w2vCap = nn.Embedding(self.vocab_size, self.feat_size)
        self.avgpool = nn.AvgPool2d(self.wh, stride=1)
        
        self.weights_init()

    def forward(self,x,caption,mask):
        self.batch_size=caption.shape[0]
        Cap_emb=self.w2vCap(caption)
        
        length=(mask.sum(dim=1)).long()
        self.maxlen = length.tolist()
        
        x = self.avgpool(x)
        x = x.view(x.shape[0],-1)
        Lh, Lc = self.init_hidden_state(x)
        
        predictions = torch.zeros(self.batch_size, max(self.maxlen), self.vocab_size).to(device)
        length = (mask.sum(dim=1)).long()
        self.maxlen = length.tolist()

        for T in range(max(self.maxlen)):
            batch_size_t = sum([l > T for l in self.maxlen])
            if T < max(self.maxlen) - 1:
                Lh, Lc = self.LanguageLSTM(Cap_emb[:batch_size_t,T,:],
                                           (Lh[:batch_size_t], Lc[:batch_size_t]))
                oword = self.classfier(self.Dropout05(Lh))
                predictions[:batch_size_t, T, :] = oword

        return predictions

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0)
            self.w2vCap.weight.data.uniform_(self.minval, self.maxval)

    def init_hidden_state(self,init_feats):
        h = self.f2h(init_feats)  # (batch_size, decoder_dim)
        c = self.f2c(init_feats)
        return h, c

    def sample(self,feats,sample_max):
        batch_size = feats.shape[0]
        seq_length = 16
        feats = self.avgpool(feats)
        feats = feats.view(feats.shape[0],-1)
        h, c = self.init_hidden_state(feats)
        it = torch.zeros(batch_size, dtype=torch.long).to(device)
        seq = torch.zeros(batch_size, seq_length, dtype=torch.long).to(device)
        seqLogprobs = torch.zeros(batch_size, seq_length).to(device)
        for T in range(seq_length+1):
            word_emb = self.w2vCap(it)
            h, c = self.LanguageLSTM(word_emb, (h,c))
            o = self.classfier(h)
            logprobs = F.log_softmax(o, dim=1)
            if T == seq_length :
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                prob_prev = torch.exp(logprobs.data).cpu()
                it = torch.multinomial(prob_prev, 1).to(device)
                sampleLogprobs = logprobs.gather(1, it)
                it = it.view(-1).long()
                # stop when all finished
            if T == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,T] = it #seq[t] the input of t+2 time step
            seqLogprobs[:,T] = sampleLogprobs.view(-1)
            if unfinished.sum() == 0:
                break   

        return seq, seqLogprobs     


class VSE_CAP(nn.Module):
    def __init__(self, feat_size, hidden_size, vocab_size, sptial_size, channel_size, out_channel):
        super(VSE_CAP, self).__init__()

        self.CNN = resnet101(pretrained=True)
        self.CBAM_SCA = CBAM_SCA(feat_size, hidden_size, vocab_size, sptial_size,
                                 channel_size, out_channel)
        
    def forward(self, x, caption, mask):
        x = self.CNN(x)
        predictions = self.CBAM_SCA(x, caption, mask)
        
        return predictions


