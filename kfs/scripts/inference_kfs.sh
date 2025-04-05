#!/bin/bash
MASTER_PORT=$((25000 + $RANDOM % 100))
# Feel free to modify these. Take mevis valid benchmark as an example.

DIR_PATH=''

MODEL_PATH=$DIR_PATH/outputs/kfs_model
SAVE_DIR=$DIR_PATH/outputs


CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port=$MASTER_PORT eval_ds.py \
   --version="$MODEL_PATH" \
   --log_base_dir="$SAVE_DIR" \
   --exp_name="kfs_inference_mevis" \
   --balance_sample \
   --dataset="rvos" \
   --sample_rates="13" \
   --val_dataset "mevis_test" \
   --eval_only 

ORI_PATH=$SAVE_DIR/kfs_inference_mevis/mevis_test/Anootations/0
python utils/extract_key_frames.py \
   --original_path="$ORI_PATH" \
   --save_path="$SAVE_DIR/kfs_select_inference/mevis.json"