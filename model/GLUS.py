"""
Adapted from https://github.com/dvlab-research/LISA/blob/main/model/LISA.py
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN, dict_to_cuda)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms

from utils.contrastive_loss import ContrastiveLoss

from PIL import Image

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class GlusMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GlusMetaModel, self).__init__(config)

        self.config = config
        self.config.mask_threshold = 0.0
        self.config.max_hole_area = 0.0
        self.config.max_sprinkle_area = 0.0
        self.sam_config = kwargs.get("sam_config", None)
        
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_glus_modules(self.config)

    def initialize_glus_modules(self, config):
        # SAM
        self.visual_model = build_sam2(config_file=self.sam_config, ckpt_path=self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.sam_mask_decoder.train()
            for param in self.visual_model.sam_mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
            

class GlusModel(GlusMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(GlusModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False



class GLUSForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.contrastive_loss_weight = kwargs.pop("contrastive_loss_weight", None)
        
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = GlusModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
        self.not_use_mem_bank = kwargs.pop("not_use_mem_bank", False)
        self.image_features_num = kwargs.pop("image_features_num", 63)
        use_contrastive_loss = kwargs.pop("use_contrastive_loss", False)
        # self.collate_fn_args = kwargs.pop("collate_fn_args", None)
        self.feat_sizes = [(256, 256), (128, 128), (64, 64)]
        self.transform = SAM2Transforms(512, mask_threshold=0.0) # 512 should be of no use
        if self.not_use_mem_bank:
            self.memory_bank_list = None
        else:
            self.memory_bank_list = {}
        self.num_maskmem = kwargs.pop("num_maskmem", 7) # and we use 6 as mem_cat
        self.super_resolution_for_sam_mask_encoder = (1024, 1024)
        
        if use_contrastive_loss:
            self.contrastive_loss = ContrastiveLoss(kwargs.pop("base_image_dir", None), taw=0.07)
        else:
            self.contrastive_loss = None

    def get_visual_embs(self, pixel_values: torch.FloatTensor, frame_id: int, bt_sz: List[int], mem_stride = 1):
        
        with torch.no_grad():
            batch_size = pixel_values.shape[0]
            assert batch_size == len(bt_sz)
            sum_bt_sz = np.concatenate(([0], np.cumsum(bt_sz)))
            vision_feats = []
            vision_pos_embeds = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                backbone_out = self.model.visual_model.forward_image(pixel_values[i].unsqueeze(0))
                _, vision_feats_per, vision_pos_embeds_per, _ = self.model.visual_model._prepare_backbone_features(backbone_out)
                vision_feats.append(vision_feats_per)
                vision_pos_embeds.append(vision_pos_embeds_per)
            torch.cuda.empty_cache()
            # vision feats: batch_size * 3 * featnum^2 * 1 * dim
            
            vision_feats_list, vision_pos_embeds_list = [], []
            diff_size_feature_num = len(vision_feats[0])
            for i in range(batch_size):
                assert len(vision_feats[i]) == diff_size_feature_num
            for i in range(diff_size_feature_num):
                vision_feats_list.append(
                    torch.cat([
                        vision_feats[bz_id][i].repeat(1, bt_sz[bz_id], 1) 
                        for bz_id in range(batch_size)], dim=1)
                    .to(pixel_values.dtype)
                    )
                vision_pos_embeds_list.append(
                    torch.cat([
                        vision_pos_embeds[bz_id][i].repeat(1, bt_sz[bz_id], 1) 
                        for bz_id in range(batch_size)], dim=1)
                    .to(pixel_values.dtype)
                    )
            
            vision_feats = vision_feats_list
            vision_pos_embeds = vision_pos_embeds_list
            
            new_bs = sum(bt_sz)
            assert vision_feats[0].shape[1] == new_bs
            
            torch.cuda.empty_cache()
            
            if self.not_use_mem_bank or len(self.memory_bank_list) == 0:
                vision_feats[-1] = vision_feats[-1] + self.model.visual_model.no_mem_embed
            else:
                # SAM2-like memory bank
                for i in range(batch_size):
                    
                    bs_l = sum_bt_sz[i]
                    bs_r = sum_bt_sz[i + 1]
                    to_cat_memory, to_cat_memory_pos = [], []
                    
                    r = mem_stride
                    for t_pos in range(0, self.num_maskmem):
                        t_rel = self.num_maskmem - t_pos
                        if t_rel == 1:
                            prev_frame_idx = frame_id - t_rel
                        else:
                            prev_frame_idx = ((frame_id - 2) // r) * r
                            prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                        if prev_frame_idx < 0:
                            continue
                        
                        element = self.memory_bank_list.get(prev_frame_idx, None)
                        if element is None:
                            print(f"Alert: Element at postion {prev_frame_idx} not in memory bank, skip it.")
                            continue
                        element = element[i]
                        
                        to_cat_memory.append(element[0].flatten(2).permute(2, 0, 1))
                        maskmem_enc = element[1].flatten(2).permute(2, 0, 1)
                        to_cat_memory_pos.append(maskmem_enc)
                        
                    if len(to_cat_memory) == 0:
                        
                        vision_feats[-1] = vision_feats[-1] + self.model.visual_model.no_mem_embed
                        
                    else:
                        
                        memory = torch.cat(to_cat_memory, dim=0)
                        memory_pos = torch.cat(to_cat_memory_pos, dim=0)
                        
                        vision_feats[-1][:, bs_l:bs_r] = self.model.visual_model.memory_attention(
                            curr=[vision_feats[-1][:, bs_l:bs_r]],
                            curr_pos=[vision_pos_embeds[-1][:, bs_l:bs_r]],
                            memory=memory,
                            memory_pos=memory_pos,
                            num_obj_ptr_tokens=0,
                        )
            
            feats = [
                feat.permute(1, 2, 0).view(new_bs, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], self.feat_sizes[::-1])
            ][::-1]
            
            image_embed = feats[-1]
            high_res_feats = feats[:-1]
            
        return image_embed, high_res_feats, vision_feats
    
    def clear_mem_bank(self):
        self.memory_bank_list.clear()

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.generate_masks(**kwargs)
    
    def generate_masks(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        rel_pos_list: List[int],
        sampled_str_ids_list: List[List[str]],
        sampled_frames_list: List[List[str]],
        inference: bool = False,
        context_frame_num: int = 4,
        question_frame_num: int = 4,
        **kwargs,
    ):
        
        frame_num = images.shape[1]
        assert frame_num == context_frame_num + question_frame_num
        
        image_features_num = self.image_features_num
        
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], context_frame_num * image_features_num)).bool().cuda(), seg_token_mask],
            dim=1,
        )
        
        per_batch_mask_length = seg_token_mask.shape[1] + question_frame_num * image_features_num
        
        _seg_token_mask = torch.zeros((seg_token_mask.shape[0], per_batch_mask_length)).bool().cuda()
        
        for i in range(seg_token_mask.shape[0]):
            curr_seg_token_mask = seg_token_mask[i]
            col_idx = 0
            for j in range(seg_token_mask.shape[1]):
                if seg_token_mask[i, j] == 1:
                    col_idx += image_features_num
                    _seg_token_mask[i, col_idx] = 1
                else:
                    _seg_token_mask[i, col_idx] = 0
                col_idx += 1
            if col_idx != per_batch_mask_length:
                raise ValueError('{} != {}'.format(col_idx, per_batch_mask_length))
            
        seg_token_mask = _seg_token_mask

        if inference:
            raise NotImplementedError("Using inference mode in 'forward' is not preferred. Please use 'evaluate' instead.")
        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .repeat(end_i - start_i, 1, 1, 1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)
            torch.cuda.empty_cache()
            
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
                specific_ce_loss=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )
        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        normalized_embeddings = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
            normalized_embeddings.append(pred_embeddings[start_i:end_i])
            # normalized_embeddings.append(nn.functional.normalize(pred_embeddings[start_i:end_i]))
        pred_embeddings = pred_embeddings_
        
        
        contrastive_loss = None
        if self.contrastive_loss is not None:
            for bs in range(len(normalized_embeddings)):
                str_ids_len = len(sampled_str_ids_list[bs])
                frame_list_len = len(sampled_frames_list[bs])
                for i in range(str_ids_len):
                    curr_str_id = sampled_str_ids_list[bs][i]
                    for j in range(frame_list_len):
                        curr_frame = sampled_frames_list[bs][j]
                        self.contrastive_loss.update(curr_str_id, curr_frame, normalized_embeddings[bs][i * frame_list_len + j].clone().detach())
            
            contrastive_loss = 0.0
            contrastive_loss_cnt = 0
            for bs in range(len(normalized_embeddings)):
                str_ids_len = len(sampled_str_ids_list[bs])
                frame_list_len = len(sampled_frames_list[bs])
                for i in range(str_ids_len):
                    curr_str_id = sampled_str_ids_list[bs][i]
                    for j in range(frame_list_len):
                        curr_frame = sampled_frames_list[bs][j]
                        curr_dt = {}
                        curr_dt['dataset'], curr_dt['video'], curr_dt['exp_id'] = curr_str_id.split('_', 2)
                        curr_dt['frame_id'] = curr_frame
                        curr_dt['seg_token'] = normalized_embeddings[bs][i * frame_list_len + j]
                        curr_contrastive_loss = self.contrastive_loss.get_loss(curr_dt)
                        if curr_contrastive_loss != None:
                            contrastive_loss += curr_contrastive_loss
                            contrastive_loss_cnt += 1
            
            if contrastive_loss_cnt > 1:
                contrastive_loss /= contrastive_loss_cnt
                contrastive_loss *= self.contrastive_loss_weight
            else:
                contrastive_loss = None
        
        if self.not_use_mem_bank == False and rel_pos_list[0] == 0:
            self.clear_mem_bank()
        
        bt_sz = [pred_embeddings[_].shape[0] // question_frame_num for _ in range(len(pred_embeddings))]
        sum_bt_sz = np.concatenate(([0], np.cumsum(bt_sz)))
        
        pred_masks = []
        
        for j in range(question_frame_num):
            
            image_embed, high_res_feats, vision_feats = self.get_visual_embs(images[:, j + context_frame_num], rel_pos_list[j], bt_sz) # 
            new_bs = image_embed.shape[0]
            assert new_bs == sum(bt_sz)
            
            diff_size_feature_num = len(vision_feats)
            assert diff_size_feature_num == len(self.feat_sizes)
            
            # from now on, we use new_bs = sum(bt_sz), e.g. 3 + 3
            pred_cur_masks = []
            
            for i in range(len(pred_embeddings)):
                cur_index = [j + question_frame_num * num for num in range(pred_embeddings[i].shape[0] // question_frame_num)] 
                # segmentation token for each frame in each round of conversations, and conversation is the cause of interval.
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.sam_prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i][cur_index].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                
                idx_range_new_bs = list(range(sum_bt_sz[i], sum_bt_sz[i + 1]))
                
                low_res_masks, iou_predictions, sam_output_tokens, object_score_logits = self.model.visual_model.sam_mask_decoder(
                    image_embeddings=image_embed[idx_range_new_bs],
                    image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    high_res_features=[high_res_feats[_][idx_range_new_bs] for _ in range(diff_size_feature_num - 1)],
                    repeat_image=False,
                )
                pred_mask = self.transform.postprocess_masks(
                    low_res_masks,
                    orig_hw=label_list[i].shape,
                )
                
                if self.not_use_mem_bank == False:
                
                    vision_feats_curr_batch = [vision_feats[fid][:, idx_range_new_bs] for fid in range(diff_size_feature_num)]
                    
                    maskmem_features, maskmem_pos_enc = self.model.visual_model._encode_new_memory(
                        current_vision_feats=vision_feats_curr_batch,
                        feat_sizes=self.feat_sizes,
                        pred_masks_high_res=self.transform.postprocess_masks(
                            low_res_masks,
                            orig_hw=self.super_resolution_for_sam_mask_encoder
                        ).to(low_res_masks.dtype), 
                        # curr_batch(among [0,1]), exp_num(bs) * multi_mask_num(1) * H * W
                        is_mask_from_pts=False,
                    )
                    
                    cur_frame_list = self.memory_bank_list.get(rel_pos_list[j], None)
                    if cur_frame_list is None:
                        self.memory_bank_list[rel_pos_list[j]] = []
                    self.memory_bank_list[rel_pos_list[j]].append([
                        maskmem_features.detach(), 
                        maskmem_pos_enc[0].detach(),
                    ])
                
                pred_cur_masks.append(pred_mask[:, 0])
                
            pred_masks.append(pred_cur_masks)
            
        # pred_masks: q_frame_num, bs, expression_num * mask_size
        model_output = output
        gt_masks = masks_list
        
        output = model_output.logits
        
        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        
        for batch_idx in range(len(gt_masks)):
            for frame_id in range(question_frame_num):
                
                gt_mask = gt_masks[batch_idx][:, frame_id]
                pred_mask = pred_masks[frame_id][batch_idx]
                
                assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
                ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                    gt_mask.shape, pred_mask.shape
                )
                mask_bce_loss += (
                    sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss
        if contrastive_loss != None:
            loss += contrastive_loss
        
        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
            "contrastive_loss": contrastive_loss,
        }


    
    #todo evaluate
    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        rel_pos_list,
        mask_clips_list,
        max_new_tokens=512,
        tokenizer=None,
        context_frame_num=4,
        question_frame_num=4,
        mem_stride=1,
        decode_iter=False,
    ):
        
        with torch.no_grad():
            
            frame_num = images.shape[1]
            mask_num = 0
            if mask_clips_list is not None:
                mask_num = mask_clips_list[0].shape[1]
            assert frame_num == context_frame_num + question_frame_num
            image_features_num = self.image_features_num
            
            if mask_clips_list is None:
                outputs = self.generate(
                    images=images_clip[0],
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
            else:
                assert len(mask_clips_list) == 1 and mask_clips_list[0].shape[0] == 1
                images_list = []
                images_label_list = []
                images_list.append(images_clip[0][:context_frame_num])
                images_label_list.append(torch.zeros(context_frame_num))
                for i in range(question_frame_num):
                    if i == 0:
                        images_list.append(images_clip[0][context_frame_num + i].unsqueeze(0))
                        images_label_list.append(torch.zeros(1))
                    else:
                        images_list.append(images_clip[0][context_frame_num + i].unsqueeze(0))
                        images_label_list.append(torch.zeros(1))
                        
                outputs = self.generate(
                    images=torch.cat(images_list, dim=0),
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences
            
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], context_frame_num * image_features_num)).bool().cuda(), seg_token_mask],
                dim=1,
            )
            
            per_batch_mask_length = seg_token_mask.shape[1] + question_frame_num * image_features_num
            
            num_add_mask = 0
        
            _seg_token_mask = torch.zeros((seg_token_mask.shape[0], per_batch_mask_length)).bool().cuda()
            
            for i in range(seg_token_mask.shape[0]):
                curr_seg_token_mask = seg_token_mask[i]
                col_idx = 0
                for j in range(seg_token_mask.shape[1]):
                    if seg_token_mask[i, j] == 1:
                        num_add_mask += 1
                        col_idx += image_features_num
                        _seg_token_mask[i, col_idx] = 1
                    else:
                        _seg_token_mask[i, col_idx] = 0
                    col_idx += 1
                    
            if col_idx != per_batch_mask_length:
                print('{} != {}'.format(col_idx, per_batch_mask_length))
                return output_ids, None, None
                
            seg_token_mask = _seg_token_mask
            

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]
            
            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )
            # seg_token_offset = seg_token_offset[offset]

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_
            
            if decode_iter:
                bt_sz = [pred_embeddings[_].shape[0] // seg_token_counts[_].item() for _ in range(len(pred_embeddings))]
            else:
                bt_sz = [pred_embeddings[_].shape[0] // question_frame_num for _ in range(len(pred_embeddings))]
            
            sum_bt_sz = np.concatenate(([0], np.cumsum(bt_sz)))
            
            pred_masks = []
            iou_scores = []
            
            if decode_iter:
                iter_j = [seg_token_counts[0] - 1]
            else:
                iter_j = list(range(question_frame_num))
                
            for j in iter_j:
                
                if self.not_use_mem_bank is False and rel_pos_list[j] == 0:
                    self.clear_mem_bank()
                
                image_embed, high_res_feats, vision_feats = self.get_visual_embs(images[:, j + context_frame_num], rel_pos_list[j], bt_sz, mem_stride=mem_stride) # 
                new_bs = image_embed.shape[0]
                assert new_bs == sum(bt_sz)
                
                diff_size_feature_num = len(vision_feats)
                assert diff_size_feature_num == len(self.feat_sizes)
                
                # from now on, we use new_bs = sum(bt_sz), e.g. 3 + 3
                pred_cur_masks = []
                iou_cur_scores = []
                
                for i in range(len(pred_embeddings)):
                    if decode_iter:
                        frame_num_to_devide = seg_token_counts[i].item()
                    else:
                        frame_num_to_devide = question_frame_num
                        
                    cur_index = [j + frame_num_to_devide * num for num in range(pred_embeddings[i].shape[0] // frame_num_to_devide)] 
                    if len(pred_embeddings[i][cur_index].shape) == 1:
                        text_embeds = pred_embeddings[i][cur_index].unsqueeze(0).unsqueeze(1)
                    else:
                        text_embeds = pred_embeddings[i][cur_index].unsqueeze(1)
                        
                    # segmentation token for each frame in each round of conversations, and conversation is the cause of interval.
                    (
                        sparse_embeddings,
                        dense_embeddings,
                    ) = self.model.visual_model.sam_prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                        text_embeds=text_embeds,
                    )
                    sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                    
                    idx_range_new_bs = list(range(sum_bt_sz[i], sum_bt_sz[i + 1]))
                    
                    low_res_masks, iou_predictions, sam_output_tokens, object_score_logits = self.model.visual_model.sam_mask_decoder(
                        image_embeddings=image_embed[idx_range_new_bs],
                        image_pe=self.model.visual_model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        high_res_features=[high_res_feats[_][idx_range_new_bs] for _ in range(diff_size_feature_num - 1)],
                        repeat_image=False,
                    )
                    pred_mask = self.transform.postprocess_masks(
                        low_res_masks,
                        orig_hw=original_size_list[i],
                    )
                    
                    if self.not_use_mem_bank is False:
                    
                        vision_feats_curr_batch = [vision_feats[fid][:, idx_range_new_bs] for fid in range(diff_size_feature_num)]
                        
                        maskmem_features, maskmem_pos_enc = self.model.visual_model._encode_new_memory(
                            current_vision_feats=vision_feats_curr_batch,
                            feat_sizes=self.feat_sizes,
                            pred_masks_high_res=self.transform.postprocess_masks(
                                low_res_masks,
                                orig_hw=self.super_resolution_for_sam_mask_encoder,
                            ).to(low_res_masks.dtype), 
                            # curr_batch(among [0,1]), exp_num(bs) * multi_mask_num(1) * H * W
                            is_mask_from_pts=False,
                        )
                        
                        cur_frame_list = self.memory_bank_list.get(rel_pos_list[j], None)
                        if cur_frame_list is None:
                            self.memory_bank_list[rel_pos_list[j]] = []
                        self.memory_bank_list[rel_pos_list[j]].append([
                            maskmem_features.detach(), 
                            maskmem_pos_enc[0].detach(),
                        ])
                    
                    pred_cur_masks.append(pred_mask[:, 0])
                pred_masks.append(pred_cur_masks)
        
        return output_ids, pred_masks, outputs
