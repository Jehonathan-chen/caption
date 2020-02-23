# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:01:31 2019

@author: yuxuan
"""

import sys
sys.path.append("cider")
sys.path.append("coco_caption")
from pyciderevalcap.ciderD.ciderD import CiderD
from pycocoevalcap.bleu.bleu import Bleu
from collections import OrderedDict
import torch
import numpy as np
import json

CiderD_scorer = None
Bleu_scorer = None

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)

vocab_f='coco_vocab.json'

with open(vocab_f) as f:
    vocab=json.load(f)
    vocab=vocab['itow']


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        if arr[i] != 0:
          out += vocab[str(arr[i])] + ' '
        elif arr[i] == 0:
            out +='<eos>'
            break
    return out.strip()


def get_self_critical_reward(model,feats,data_gts,gen_result):
    #print('gen_result: ',gen_result)
    #print(data_gts)
    batch_size = gen_result.size(0)
    seq_per_img = batch_size // len(data_gts)
    cider_reward_weight=0.7
    bleu_reward_weight=0.3
    
    init_scorer('-words')
    
    model.eval()
    with torch.no_grad():
        greedy_res,_ = model.sample(feats,sample_max=1)
        #print(greedy_res)
    model.train()
    
    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]
    #print(res)
    
    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    #print('res_:',res_)
    res__ = {i: res[i] for i in range(2 * batch_size)}
    #print('res__:',res__)
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    #print('gts: ',gts)
    _, cider_scores = CiderD_scorer.compute_score(gts, res_)
    
    #print(gts)
    print('Cider scores:', _)
    _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
    bleu_scores = np.array(bleu_scores[3])
    print('Bleu scores:', _[3])
    
    scores = cider_reward_weight * cider_scores + bleu_reward_weight * bleu_scores
    #print(scores[batch_size:])
    scores = scores[:batch_size] - scores[batch_size:]
    #print(scores)
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)
    return rewards
    
    
    
    
    



