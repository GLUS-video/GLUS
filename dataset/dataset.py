import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
import json
from pycocotools import mask 

from model.llava import conversation as conversation_lib
from model.llava.constants import (DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX)
from model.llava.mm_utils import tokenizer_image_token
from sam2.utils.transforms import SAM2Transforms

from .conversation import get_default_conv_template

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)

from .refer_video_seg_dataset import ReferVideoSegDataset


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    sampled_str_ids_list = []
    sampled_frames_list = []
    cnt = 0
    inferences = []
    for (
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
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(image_clips)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resizes)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        sampled_str_ids_list.append(sampled_str_ids)
        sampled_frames_list.append(sampled_frames)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
    # print(conversation_list)
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "sampled_str_ids_list": sampled_str_ids_list,
        "sampled_frames_list": sampled_frames_list,
    }


class HybridDataset(torch.utils.data.Dataset):
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
        dataset="refer_video_seg",
        sample_rate=[1],
        refer_video_seg_data="mevis",
        explanatory=0.1,
        context_frame_num=4,
        question_frame_num=4,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.context_frame_num = context_frame_num
        self.question_frame_num = question_frame_num

        self.datasets = dataset.split("||")

        self.all_datasets = []
        for dataset in self.datasets:
            
            if dataset == "refer_video_seg":
                self.all_datasets.append(
                    ReferVideoSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_video_seg_data,
                        context_frame_num,
                        question_frame_num,
                        sample_rate,
                    )
                )
            else:
                raise NotImplementedError("Simply use refer_video_seg.")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        
        ind = np.random.choice(list(range(len(self.datasets))))
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference
    
class ValDataset(torch.utils.data.Dataset):
    '''
    Alert: Validation dataset is not used in GLUS.
    '''