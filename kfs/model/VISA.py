from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN

from .univi.model.language_model.llama import ChatUniViLlamaForCausalLM, ChatUniViLlamaModel

from .segment_anything import build_sam_vit_h
from model.univi.constants import IMAGE_TOKEN_INDEX

import time


class VisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(VisaMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            # self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            # self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):

        # Projection layer
        in_dim = config.hidden_size
        h_dim = config.out_dim
        out_dim = 1 #equal to num mask tokens
        # h_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, out_dim),
            nn.Dropout(0.0),
            # nn.Sigmoid(),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class VisaModel(VisaMetaModel, ChatUniViLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(VisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class VISAForCausalLM(ChatUniViLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get("vision_tower", "openai/clip-vit-large-patch14")
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.iou_loss_weight = kwargs.pop("iou_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.score_token_idx = kwargs.pop("score_token_idx")
        super().__init__(config)

        self.model = VisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        scores_list: List[torch.Tensor],
        conversation_list: List[str], 
        num_frame_list: List[int],
        num_conv_list: List[int],
        inference: bool = False,
        **kwargs,
    ):
        assert batch_size == len(offset) - 1
        for batch_idx in range(batch_size):
            assert num_conv_list[batch_idx] == offset[batch_idx + 1] - offset[batch_idx]

        if inference:
            length = input_ids.shape[0]
            assert len(images_clip) == 1, f'Inference only supports one video, but got {len(images_clip)} videos.'
            images_clip = [
                images_clip[0].unsqueeze(0).expand(length, -1, -1, -1, -1).contiguous().flatten(0,1)
            ]

            output_i = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                output_hidden_states=True,
            )
            torch.cuda.empty_cache()

            output_hidden_states = output_i.hidden_states
            output = None

            num_image_ori_token = (input_ids[0] == IMAGE_TOKEN_INDEX).sum()
            assert all(
                [
                    (input_ids[i] == IMAGE_TOKEN_INDEX).sum() == num_image_ori_token for i in range(length)
                ]
            )
            token_add = 111 * num_image_ori_token
            
            score_token_mask = input_ids[:, 1:] == self.score_token_idx
            score_token_mask = torch.cat([score_token_mask,  torch.zeros((score_token_mask.shape[0], 1)).bool().cuda(), ], dim=1, )
            score_token_mask = torch.cat([torch.zeros((score_token_mask.shape[0], token_add)).bool().cuda(), score_token_mask], dim=1, )
            all_conv_score_token_num = score_token_mask.sum(dim=1).tolist()

        else:
            images_clip_list = []
            for batch_idx in range(batch_size):
                bs_conv_num = num_conv_list[batch_idx]
                images_clip_i = images_clip[batch_idx].unsqueeze(0).expand(bs_conv_num, -1, -1, -1, -1).contiguous()
                images_clip_list.append(images_clip_i)
            images_clip_list = [i.flatten(0,1) for i in images_clip_list]

            output = super().forward(
                images=images_clip_list,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

            score_token_mask = output.labels[..., 1:] == self.score_token_idx
            score_token_mask = torch.cat([score_token_mask,  torch.zeros((score_token_mask.shape[0], 1), device=output.labels.device).bool(), ], dim=1, )
            all_conv_score_token_num = score_token_mask.sum(dim=1).tolist()

        assert len(self.model.text_hidden_fcs) == 1
        
        pred_embeddings = self.model.text_hidden_fcs[0](output_hidden_states[-1][score_token_mask])
        score_token_counts = score_token_mask.int().sum(-1)  # [bs, ]
        score_token_offset = score_token_counts.cumsum(-1)
        score_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), score_token_offset], dim=0
        )
        score_token_offset = score_token_offset[offset]
        pred_embeddings_ = []
        for i in range(len(score_token_offset) - 1):
            start_i, end_i = score_token_offset[i], score_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_
        
        assert len(pred_embeddings) == batch_size

        model_output = output

        if inference:
            return {
                "pred_ious": pred_embeddings
            }
            return

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        iou_loss = 0.
        #iou loss L1 loss between score_list and pred_embeddings
        for i in range(batch_size):
            iou_loss += F.l1_loss(pred_embeddings[i], scores_list[i].squeeze(1))
        iou_loss = iou_loss * self.iou_loss_weight / (batch_size + 1e-8)

        loss = ce_loss + iou_loss 
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "iou_loss": iou_loss,
        }

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("This method is not implemented.")