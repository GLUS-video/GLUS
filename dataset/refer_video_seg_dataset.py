"""
Ref-YoutubeVOS data loader
"""
from pathlib import Path
import sys
import math
from enum import IntEnum, unique
from typing import List, Tuple, Union
import copy

import torch
from torch.utils.data import Dataset

import os
from PIL import Image
import json
import numpy as np
import random
from tqdm import tqdm

import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask 
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from sam2.utils.transforms import SAM2Transforms

from utils.utils import ANSWER_LIST, SHORT_QUESTION_LIST, CONTEXT_INFO_LIST
from .mevis import load_mevis_json
from .refyoutube_vos import load_refyoutube_json
from .davis17 import load_davis17_json
from .revos import load_revos_json
from .lvvis import load_lvvis_json

    
class ReferVideoSegDataset(Dataset):
    """
    A dataset class for Mevis only

    """
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        refer_video_seg_data="mevis",
        context_frame_num = 4,
        question_frame_num = 4,
        sample_rate = [1],
        is_train = True,
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = SAM2Transforms(image_size, mask_threshold=0.0)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.context_info_list = CONTEXT_INFO_LIST
        
        self.context_frame_num = context_frame_num
        self.question_frame_num = question_frame_num
        self.is_train = is_train

        DATA_DIR = os.path.join(base_image_dir, "refer_seg")
        self.refer_video_seg_ds_list = refer_video_seg_data.split(
            "||"
        )
        
        self.refer_video_seg_data = {}
        
        for ds in self.refer_video_seg_ds_list: 
            
            refer_video_seg_ds = {}
            
            if ds == "mevis":
                refer_video_seg_ds["vid_list"], refer_video_seg_ds["metas"], refer_video_seg_ds["mask_dict"] = load_mevis_json(self.base_image_dir, self.is_train)                
            elif ds == "revos":
                refer_video_seg_ds["vid_list"], refer_video_seg_ds["metas"], refer_video_seg_ds["mask_dict"] = load_revos_json(self.base_image_dir)
            elif ds == "lvvis":
                refer_video_seg_ds["vid_list"], refer_video_seg_ds["metas"], refer_video_seg_ds["mask_dict"] = load_lvvis_json(self.base_image_dir)
            elif ds == "refyoutube_vos":
                refer_video_seg_ds["vid_list"], refer_video_seg_ds["metas"], refer_video_seg_ds["vid2masks"] = load_refyoutube_json(self.base_image_dir)                    
            elif ds == "davis17":
                refer_video_seg_ds["vid_list"], refer_video_seg_ds["metas"], refer_video_seg_ds["vid2masks"] = load_davis17_json(self.base_image_dir)      
            else:
                raise NotImplementedError(f"Unknown refer_video_seg_data: {ds}")
            
            vid2refs = {}
            for ref in refer_video_seg_ds["metas"]:
                video_id = ref["video"]
                vid2refs[video_id] = vid2refs.get(video_id, []) + [
                    ref,
                ]
                refer_video_seg_ds["vid2refs"] = vid2refs
            
            self.refer_video_seg_data[ds] = refer_video_seg_ds
        
        # Modify the sampling ratio here
        self.rd_whole_list = {}
        for dname, dprob in zip(self.refer_video_seg_ds_list, sample_rate):
            self.rd_whole_list[dname] = dprob
            
        print("rd_whole_list", self.rd_whole_list)
        
    def __len__(self):
        return len(self.refer_video_seg_data)
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def __getitem__(self, idx):
        
        rd_whole_list = self.rd_whole_list
        rd_list = {}
        for ds in self.refer_video_seg_ds_list:
            rd_list[ds] = rd_whole_list[ds]
        rd_dss = list(rd_list.keys())
        rd_exps = list(rd_list.values())
        
        ds = random.choices(rd_dss, weights=rd_exps, k=1)[0]
        refer_video_seg_ds = self.refer_video_seg_data[ds]
        vid_list = refer_video_seg_ds['vid_list']
        
        metas = refer_video_seg_ds['metas']
        if ds == "mevis" or ds == "revos" or ds == "lvvis":
            mask_dict = refer_video_seg_ds['mask_dict']
        elif ds == "refyoutube_vos" or ds == "davis17":
            vid2masks = refer_video_seg_ds['vid2masks']
        vid2refs = refer_video_seg_ds['vid2refs']
        
        while True:
            idx = random.randint(0, len(vid_list) - 1)
            image_path = None
            video_id = vid_list[idx]
            if video_id not in vid2refs:
                continue
            refs = vid2refs[video_id]
            video_meta = refs[0]
            if video_meta['length'] >= max(self.context_frame_num, self.question_frame_num):
                break
        
        if len(refs) == 0:
            return self.__getitem__(0)

        sents = []
        ann_ids = []
        for ref in refs:
            sent = ref["exp"]
            text = sent[:-1] if sent.endswith('.') else sent
            sents.append(text)
            ann_ids.append(ref["anno_id"])
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        sampled_classes = sampled_sents
        
        str_ids = [ref['str_id'] for ref in refs]
        sampled_str_ids = [str_ids[ind] for ind in sampled_inds]
        
        #sample index
        images = []
        image_clips = []
        resizes = []
        video_length = video_meta["length"]
        video_paths = video_meta["file_names"]
        
        assert video_length >= self.question_frame_num
        sampled_start = random.randint(0, video_length - self.question_frame_num)
        sampled_question_indices = list(range(sampled_start, sampled_start + self.question_frame_num))
        
        indices = np.linspace(0, video_length, self.context_frame_num + 1, dtype=int)
        sampled_context_indices = [np.random.choice(range(indices[i], indices[i + 1])) for i in range(self.context_frame_num)]
        
        sampled_indices = sampled_context_indices + sampled_question_indices
        video_paths = [video_paths[i] for i in sampled_indices]
        
        sampled_frames = [video_meta['frames'][ind] for ind in sampled_question_indices]
        
        for path in video_paths:
            
            image = cv2.imread(path)

            original_h, original_w = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            #resize image to 512, 512 
            image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            
            image = self.transform(image).contiguous()  # preprocess image for sam2
            
            resize = image.shape[:2]
            
            images.append(image)
            image_clips.append(image_clip)
            resizes.append(resize)
            
        images = torch.stack(images, dim=0)
        image_clips = torch.stack(image_clips, dim=0)
        
        questions = []
        answers = []
        for text in sampled_classes:
            text = text.strip()
            assert len(text.split("||")) == 1
            questions_per_text = []
            answers_per_text = []
            for _ in range(self.question_frame_num):
                question_template = random.choice(self.short_question_list)
                questions_per_text.append(question_template.format(class_name=text.lower()))
                
                answers_per_text.append(random.choice(self.answer_list))
            questions.append(questions_per_text)
            answers.append(answers_per_text)

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            questions_per_text = questions[i]
            answers_per_text = answers[i]
            for j in range(len(questions_per_text)):
                if j == 0:
                    context_info = random.choice(self.context_info_list)
                    conv.append_message(conv.roles[0], context_info + questions_per_text[j])
                else:
                    conv.append_message(conv.roles[0], questions_per_text[j])
                conv.append_message(conv.roles[1], answers_per_text[j])
            conversations.append(conv.get_prompt())
            i += 1
        
        
        # mask should be original size
        
        if ds == "mevis" or ds == "revos" or ds == "lvvis":
            masks = []
            for ann_ids in sampled_ann_ids:
                mask_cur = np.zeros((self.question_frame_num, original_h, original_w), dtype=np.uint8)
                for i, frame_id in enumerate(sampled_question_indices):
                    for ann_id in ann_ids:
                        try:
                            # 
                            rle = mask_dict[str(ann_id)][frame_id]
                            mask_cur[i] = mask_cur[i] + mask.decode(rle)
                            # print('Loading successfully')
                        except:
                            # print("Error Loading masks, may be invalid")
                            continue
                masks.append(mask_cur)

            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
            
        elif ds == "refyoutube_vos" or ds == "davis17":
            masks = []
            for ann_ids in sampled_ann_ids:
                mask_cur = np.zeros((self.question_frame_num, original_h, original_w), dtype=np.uint8)
                for i, frame_id in enumerate(sampled_question_indices):
                    mask_raw = Image.open(vid2masks[video_id][frame_id])
                    mask_raw = np.array(mask_raw)
                    for ann_id in ann_ids:
                        mask_id = (mask_raw == ann_id)
                        mask_cur[i] = mask_cur[i] + mask_id
                masks.append(mask_cur)
                
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks)
            
        else:
            raise ValueError(f"No such ds {ds}.")
        label = torch.ones(masks.shape[2], masks.shape[3]) * self.ignore_label
        
        return (
            image_path,
            images,
            image_clips,
            conversations,
            masks,
            label,
            resizes,
            questions,
            sampled_classes,
            sampled_str_ids,
            sampled_frames,
        )