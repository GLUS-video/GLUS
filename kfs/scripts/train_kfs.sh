#!/bin/bash

MASTER_PORT=$((25000 + $RANDOM % 100))

# Feel free to modify these
DIR_PATH=''

PATH_TO_CHECKPOINTS=$DIR_PATH/checkpoints
PATH_TO_DATA=$DIR_PATH/data/train_iou
SAVE_DIR=$DIR_PATH/outputs


CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port=$MASTER_PORT train_ds.py \
   --version="$PATH_TO_CHECKPOINTS/Chat-UniVi" \
   --log_base_dir="$SAVE_DIR" \
   --exp_name="kfs_joint_train" \
   --balance_sample \
   --dataset="rvos" \
   --sample_rates="1" \
   --univi_sample_frame_range="8,12" \
   --rvos_seg_data="mevis_train||refytvos_train" \
   --rvos_sample_ratio="4000||15000" \
   --mevis_path="$PATH_TO_DATA/mevis" \
   --youtube_path="$PATH_TO_DATA/youtube" \
   --joint_path="$PATH_TO_DATA/joint" 


cd $SAVE_DIR/kfs_joint_tran/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin

cd $DIR_PATH

python merge_lora_weights_and_save_hf_model.py \
  --version "$PATH_TO_CHECKPOINTS/Chat-UniVi" \
  --weight "$SAVE_DIR/kfs_joint_train/pytorch_model.bin" \
  --save_path="$SAVE_DIR/kfs_model"