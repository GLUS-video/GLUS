###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################

import os
import time
import argparse
import cv2
import json
import numpy as np
from pycocotools import mask as cocomask
from metrics import db_eval_iou, db_eval_boundary
import multiprocessing as mp
from PIL import Image
import csv

NUM_WOEKERS = 64

def change_type(byte):
    if isinstance(byte, bytes):
        return str(byte, encoding="utf-8")
    return json.JSONEncoder.default(type)


def eval_queue(q, rank, out_dict, mevis_pred_path):
    score_dict = {}
    while not q.empty():
        # print(q.qsize())
        vid_name, exp = q.get()

        vid = exp_dict[vid_name]

        exp_name = f'{vid_name}_{exp}'

        if not os.path.exists(f'{mevis_pred_path}/{vid_name}'):
            print(f'{vid_name} not found')
            out_dict[exp_name] = [0, 0]
            continue

        pred_0_path = f'{mevis_pred_path}/{vid_name}/{exp}/00000.png'
        pred_0 = cv2.imread(pred_0_path, cv2.IMREAD_GRAYSCALE)
        h, w = pred_0.shape
        vid_len = len(vid['frames'])
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        obj_ids = vid['expressions'][exp]['obj_id']
        anno_ids = vid['expressions'][exp]['anno_id']
        assert len(obj_ids) == len(anno_ids)
        
        # if len(obj_ids) == 1:
        #     continue # count multi-objects
        # if len(obj_ids) > 1:
        #     continue # count single-object

        for frame_idx, frame_name in enumerate(vid['frames']):
            for anno_id in anno_ids:
                mask_rle = mask_dict[str(anno_id)][frame_idx]
                if mask_rle:
                    gt_masks[frame_idx] += cocomask.decode(mask_rle)

            # print(gt_masks[frame_idx])
            # mask = gt_masks[frame_idx]
            # img_mask = Image.fromarray((mask * 255).astype(np.uint8))
            # save_path = f'datasets/mevis/valid_u/GTMasks/{vid_name}/{exp}/{frame_name}.png'
            # os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # img_mask.save(save_path)
            
            pred_masks[frame_idx] = cv2.imread(f'{mevis_pred_path}/{vid_name}/{exp}/{frame_name}.png', cv2.IMREAD_GRAYSCALE)

        j_list = db_eval_iou(gt_masks, pred_masks)
        f_list = db_eval_boundary(gt_masks, pred_masks)
        j = j_list.mean()
        f = f_list.mean()
        out_dict[exp_name] = [j, f]
        score_dict[exp_name] = [j_list, f_list]
        print(f'{exp_name},{j_list},{f_list}', file=open(f'{mevis_pred_path}/scores_list_new.log', 'a'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mevis_exp_path", type=str, help="path to mevis valid_u/meta_expressions.json")
    parser.add_argument("--mevis_mask_path", type=str, help="path to mevis valid_u/mask_dict.json")
    parser.add_argument("--mevis_pred_path", type=str, help="path to predicted masks")
    parser.add_argument("--save_name", type=str, default="mevis_test.json")
    args = parser.parse_args()
    queue = mp.Queue()
    exp_dict = json.load(open(args.mevis_exp_path))['videos']
    mask_dict = json.load(open(args.mevis_mask_path))

    shared_exp_dict = mp.Manager().dict(exp_dict)
    shared_mask_dict = mp.Manager().dict(mask_dict)
    output_dict = mp.Manager().dict()

    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid['expressions']:
            queue.put([vid_name, exp])

    start_time = time.time()
    processes = []
    for rank in range(NUM_WOEKERS):
        p = mp.Process(target=eval_queue, args=(queue, rank, output_dict, args.mevis_pred_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    with open(args.save_name, 'w') as f:
        json.dump(dict(output_dict), f)

    j = [output_dict[x][0] for x in output_dict]
    f = [output_dict[x][1] for x in output_dict]

    print(f'J: {np.mean(j)}')
    print(f'F: {np.mean(f)}')
    print(f'J&F: {(np.mean(j) + np.mean(f)) / 2}')

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" %(total_time))