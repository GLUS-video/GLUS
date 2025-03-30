import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity as csim
import os
import json
import numpy as np
from utils.seg_token_bank import segTokenBank
import random

class ContrastiveLoss(nn.Module):
    
    def __init__(self, base_image_dir, taw):
        super().__init__()
        
        base_image_dir = os.path.join(base_image_dir, 'mevis')
        data_split = "train"
        json_file=os.path.join(base_image_dir, data_split, 'meta_expressions.json')
        
        ann_file = json_file
        
        with open(str(ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        videos = list(subset_expressions_by_video.keys())

        metas = []
        
        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['video'] = vid
                meta['obj_id'] = np.array(sorted([int(x) for x in exp_dict['obj_id']]))
                meta['exp_id'] = exp_id
                metas.append(meta)
                
        vid2refs = {}
        vid2exp2positives = {}
        vid2exp2negatives = {}
        for ref in metas:
            video_id = ref["video"]
            vid2refs[video_id] = vid2refs.get(video_id, []) + [
                ref,
            ]
            
        for vid in videos:
            if vid not in vid2refs:
                continue
            refs = vid2refs[vid]
            exp2positives = {}
            exp2negatives = {}
            for ref in refs:
                exp2positives[ref['exp_id']] = [meta for meta in refs
                                                if np.array_equal(meta['obj_id'], ref['obj_id']) 
                                                and meta['exp_id'] != ref['exp_id']]
                
            vid2exp2positives[vid] = exp2positives
            
        self.vid2exp2positives = vid2exp2positives
        self.taw = taw
        self.seg_token_bank = segTokenBank(max_bank_size=512)
        
    def get_loss(self, curr_dt): # bank_data {'str_id', 'frame_id', 'seg_token'}
        bank = self.seg_token_bank.bank
        dataset, vid, exp_id, frame_id, token = curr_dt['dataset'], curr_dt['video'], curr_dt['exp_id'], curr_dt['frame_id'], curr_dt['seg_token']
        
        if dataset != 'mevis' or vid not in self.vid2exp2positives:
            return None
        positive_set = set((meta['video'], meta['exp_id'], curr_dt['frame_id']) for meta in self.vid2exp2positives[vid][exp_id])
        positives = [dt['seg_token'] for dt in bank
                    if (dt['video'], dt['exp_id'], dt['frame_id']) in positive_set]
        negatives = [dt['seg_token'] for dt in bank
                    if (dt['video'], dt['exp_id'], dt['frame_id']) not in positive_set 
                        and (dt['video'] != vid or dt['exp_id'] != exp_id)]
        
        if len(positives) == 0 or len(negatives) < 128:
            return None

        neg_indices = [random.randint(0, len(negatives) - 1) for _ in range(128)]
        negatives = torch.stack([negatives[ind] for ind in neg_indices], dim=0)
        pos_emb = random.choice(positives)
        
        neg_sum = torch.sum(torch.exp(csim(negatives, token.unsqueeze(0)) / self.taw))
        pos_sum = torch.exp(csim(pos_emb.unsqueeze(0), token.unsqueeze(0)) / self.taw)[0]
        contrastive_loss = -torch.log(pos_sum / (neg_sum + pos_sum))
        
        return contrastive_loss
    
    def update(self, str_id, frame_id, seg_token):
        return self.seg_token_bank.update(str_id, frame_id, seg_token)
    
    def bank_size(self):
        return self.seg_token_bank.size()
        