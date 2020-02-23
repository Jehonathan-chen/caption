# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:01:36 2019

@author: yuxuan
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from rewards import get_self_critical_reward, init_scorer


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class matching_loss(nn.Module):
    def __init__(self):
        super(matching_loss, self).__init__()
        self.sim = cosine_sim
    
    def forward(self, im, s):
    
        scores = 10 * self.sim(im, s)
        img_cap = -F.log_softmax(scores, dim=0)
        cap_img = -F.log_softmax(scores, dim=1)
        L1 = img_cap.diag()
        L2 = cap_img.diag()

        loss = L1.mean() + L2.mean()
        
        return loss


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0.2, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)
        
        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)
        #print(cost_s)
        #print(cost_im)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s2 = cost_s.max(1)[0]
            cost_im2 = cost_im.max(0)[0]

        return cost_s2.sum() + cost_im2.sum()


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LossWrapper(nn.Module):
    def __init__(self, model):
        super(LossWrapper, self).__init__()
        self.model = model
        self.rl_crit = RewardCriterion()

    def forward(self, feats, gts):
        out = {}
        gen_result, sample_logprobs = self.model.sample(feats,sample_max=0)
        #print('gen_result:',gen_result)
        #gts = [gts[_] for _ in gt_indices.tolist()]
        reward = get_self_critical_reward(self.model, feats, gts, gen_result)
        #print('reward:',reward)
        reward = torch.from_numpy(reward).float().to(gen_result.device)
        loss = self.rl_crit(sample_logprobs, gen_result.data, reward)
        out['reward'] = reward[:,0].mean()
        out['loss'] = loss
        return out

    