import os
import json

def load_davis17_json(base_image_dir, is_train=True, set_name=None):
        
    base_image_dir = os.path.join(base_image_dir, 'DAVIS17')
    data_split = "train" if is_train else "valid"
    if set_name != None:
        data_split = set_name
        
    image_root = os.path.join(base_image_dir, data_split, 'JPEGImages')
    json_file = os.path.join(base_image_dir, 'meta_expressions', data_split, 'meta_expressions.json')
    mask_root = os.path.join(base_image_dir, data_split, 'Annotations')
    num_instances_without_valid_segmentation = 0
    num_instances_valid_segmentation = 0


    ann_file = json_file
    with open(str(ann_file), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
    videos = list(subset_expressions_by_video.keys())
    
    metas = []
    vid_list = []
    vid2masks = {}
    if data_split == "train":
        
        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            if vid_len < 4:
                continue
            vid_list.append(vid)
            vid2masks[vid] = [os.path.join(mask_root, vid, vid_frames[i]+'.png') for i in range(vid_len)]
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['str_id'] = f'davis17_{vid}_{exp_id}'
                meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = [int(exp_dict['obj_id'])]
                meta['anno_id'] = [int(exp_dict['obj_id'])]
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = None
                meta['length'] = vid_len
                meta['file_names'] = [os.path.join(image_root, vid, vid_frames[i]+ '.jpg') for i in range(vid_len)]
                metas.append(meta)
    else:
        for vid in videos:
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            vid_list.append(vid)
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = None
                meta['anno_id'] = None
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = None
                meta['length'] = vid_len
                meta['file_names'] = [os.path.join(image_root,  vid, vid_frames[i]+ '.jpg') for i in range(vid_len)]
                metas.append(meta)
    
    return vid_list, metas, vid2masks