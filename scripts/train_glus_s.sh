#!/bin/bash

MASTER_PORT=$((25000 + $RANDOM % 100))

# Feel free to modify these

DIR_PATH=''

PATH_TO_CHECKPOINTS=$DIR_PATH/checkpoints
PATH_TO_DATA=$DIR_PATH/data
SAVE_DIR=$DIR_PATH/outputs

deepspeed --master_port=$MASTER_PORT train_ds.py \
  --version "$PATH_TO_CHECKPOINTS/LISA-7B-v1" \
  --dataset_dir "$PATH_TO_DATA" \
  --sam_config "sam2_hiera_l.yaml" \
  --vision_pretrained "$PATH_TO_CHECKPOINTS/sam2_hiera_large.pt" \
  --dataset "refer_video_seg" \
  --refer_video_seg_data="mevis||refyoutube_vos" \
  --sample_rates="1,1" \
  --exp_name "lisa-7b" \
  --no_eval \
  --steps_per_epoch 100 \
  --log_base_dir "$SAVE_DIR/logs" \
  --exp_name "glus_s" \
  --epochs 30 \
  --context_frame_num 4 \
  --question_frame_num 4 \
  --total_question_frame_num 4 \
  --use_contrastive_loss

# When modifying 'context_frame_num' and 'question_frame_num', dont forget to modify utils.utils at the same time.
# Make sure total_question_frame_num == question_frame_num.

# Merge lora weights and save hf model

cd $SAVE_DIR/logs/glus_s/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin

cd $DIR_PATH

python merge_lora_weights_and_save_hf_model.py \
  --version "$PATH_TO_CHECKPOINTS/LISA-7B-v1" \-
  --weight "$SAVE_DIR/logs/glus_s/pytorch_model.bin" \
  --save_path="$SAVE_DIR/model"