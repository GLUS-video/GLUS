#!/bin/bash

# Feel free to modify these. Take mevis valid_u benchmark as an example.

DIR_PATH=''

MODEL_PATH=$DIR_PATH/outputs/model
VIS_SAVE_PATH=$DIR_PATH/generated
PATH_TO_DATA=$DIR_PATH/data

NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

for (( GPU_ID=0; GPU_ID<$NUM_GPUS; GPU_ID++ ))
do
    echo "Launching process on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python inference_iter.py \
        --version "$MODEL_PATH" \
        --vis_save_path "$VIS_SAVE_PATH" \
        --dataset_dir "$PATH_TO_DATA" \
        --val_set='mevis' \
        --set_name='valid_u' \
        --context_frame_num 4 \
        --question_frame_num 4 &
done

echo "All processes started in background"
wait