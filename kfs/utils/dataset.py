import glob
import os
import os.path as osp
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib

from model.llava.mm_utils import tokenizer_image_token

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json


from .utils import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX, 
    convert2imagesplit, UNIFIED_SHORT_QUESTION_LIST, UNIFIED_LONG_QUESTION_LIST
)

from .rvos_dataset import RVOSDataset
from .random_list import get_random_list


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    num_frame_list = []
    num_conv_list = []
    scores_list = []
    for (
        image_path,
        images_clip,
        conversations,
        masks,
        label,
        questions,
        iou_scores,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)

        if images_clip.ndim == 3:
            images_clip = images_clip.unsqueeze(0)
        assert images_clip.ndim == 4
        images_clip_list.append(images_clip)
        num_frame = images_clip.shape[0]
        num_frame_list.append(num_frame)

        conversation_list.extend(conversations)
        label_list.append(label)
        num_conv_list.append(len(conversations))


        if masks.ndim == 3:  # [num_classes, H, W]
            if masks.shape[0] == 0:  # [0, H, W] -> [num_classes, 0, H, W]
                masks = torch.stack([masks, ] * len(conversations), dim=0).float()
            else: # [num_classes, H, W] -> [num_classes, 1, H, W]
                masks = masks.unsqueeze(1).float()
        assert masks.ndim == 4
        masks_list.append(masks.float())
        
        scores_list.append(iou_scores)

        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)

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

    for i in range(len(conversation_list)):
        if DEFAULT_VIDEO_TOKEN in conversation_list[i]:
            if conversation_list[i].count(DEFAULT_VIDEO_TOKEN) == 1:
                replace_video_token = DEFAULT_IMAGE_TOKEN * num_frame
                conversation_list[i] = conversation_list[i].replace(DEFAULT_VIDEO_TOKEN, replace_video_token)
            else:
                raise ValueError("num video token > 1: ", conversation_list[i].count(DEFAULT_VIDEO_TOKEN))


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

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    return {
        "image_paths": image_path_list,
        "images_clip": images_clip_list, #BS : T * 3 * H * W
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list, # [Conv*Frame*H*W, ...]
        "label_list": label_list, # [H*W, ...]
        "scores_list": scores_list,
        "offset": torch.LongTensor(offset_list), #[0, num_conv0, num_conv1, ...]
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "num_frame_list": num_frame_list,
        "num_conv_list": num_conv_list,
    }


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="rvos",
        sample_rate=[9, 3, 3, 1],
        rvos_seg_data="mevis_train||refytvos_train",
        rvos_sample_ratio='4000||15000',
        rvos_num_frames_sample_range="6,12",
        rvos_sample_policy="uniform",
        explanatory=0.1,
        balance_sample=True,
        meivs_path=None,
        youtube_path=None,
        joint_path=None,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")
        self.num_datasets = len(self.datasets)

        self.num_be_called = 0
        if balance_sample:
            self.dataset_sample_list = get_random_list(probabilities=self.sample_rate.tolist(), values=list(range(self.num_datasets)), length=samples_per_epoch)
            chatunivi_sample_range = [int(i) for i in univi_sample_frame_range.split(',')]
            chatunivi_range_length = chatunivi_sample_range[-1] - chatunivi_sample_range[0] + 1
            self.chatunivi_sample_list = get_random_list(probabilities=[float(1/chatunivi_range_length) for _ in range(chatunivi_range_length)], values=list(range(chatunivi_sample_range[0],chatunivi_sample_range[-1]+1)), length=10000)
            rvos_sample_range = [int(i) for i in rvos_num_frames_sample_range.split(',')]
            rvos_range_length = rvos_sample_range[-1] - rvos_sample_range[0] + 1
            self.rvos_sample_list = get_random_list(probabilities=[float(1/rvos_range_length) for _ in range(rvos_range_length)], values=list(range(rvos_sample_range[0],rvos_sample_range[-1]+1)), length=10000)
        else:
            self.dataset_sample_list = None
            self.chatunivi_sample_list = []
            self.rvos_sample_list = []
        

        self.all_datasets = []
        for dataset in self.datasets:
            elif dataset == "rvos":            
                self.all_datasets.append(
                    RVOSDataset(
                        tokenizer                = tokenizer,
                        vision_tower             = vision_tower,
                        samples_per_epoch        = samples_per_epoch,
                        precision                = precision,
                        image_size               = image_size,
                        num_classes_per_sample   = num_classes_per_sample,
                        num_frames_sample_range  = rvos_num_frames_sample_range,
                        rvos_sample_policy       = rvos_sample_policy,
                        rvos_seg_data            = rvos_seg_data,
                        rvos_sample_ratio        = rvos_sample_ratio,
                        rvos_sample_list         = self.rvos_sample_list,
                        mevis_path               = mevis_path,
                        youtube_path             = youtube_path,
                        joint_path               = joint_path,
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        self.num_be_called += 1
        if self.dataset_sample_list == None:
            ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        else:
            ind = self.dataset_sample_list[self.num_be_called % self.samples_per_epoch]
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference
