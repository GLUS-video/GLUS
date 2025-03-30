import torch
import torch.nn as nn
from collections import deque

class segTokenBank(nn.Module):
    def __init__(self, max_bank_size):
        super().__init__()
        self.bank = deque([])
        self.max_bank_size = max_bank_size
        
    def update(self, str_id, frame_id, seg_token):
        
        dataset, vid, exp_id = str_id.split('_', 2)
        if dataset != 'mevis':
            return 
        dt = {
            'dataset': dataset,
            'video': vid,
            'exp_id': exp_id,
            'frame_id': frame_id,
            'seg_token': seg_token,
        }
        self.bank.append(dt)
        while len(self.bank) > self.max_bank_size:
            old_token = self.bank.popleft()
            del old_token
            
    def size(self):
        return len(self.bank)
    