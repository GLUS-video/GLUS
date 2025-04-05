import json
import os


def extract_key_frame(original_path, save_path):
    list_video = os.listdir(original_path)
    print(len(list_video))
    all_results = {}
    fail_count = 0
    for video in list_video:
        video_path = os.path.join(original_path, video)
        exp_ids = os.listdir(video_path)
        for exp_id in exp_ids:
            exp_id_path = os.path.join(video_path, exp_id)
            frame_names = os.listdir(exp_id_path)
            max_iou = -1000
            max_score_frame = -1
            for frame_name in frame_names:
                frame_path = os.path.join(exp_id_path, frame_name)
                with open(frame_path, 'r') as f:
                    data = json.load(f)
                iou = float(data['iou'])
                if iou >= max_iou:
                    max_iou = iou
                    max_score_frame = int(frame_name.split('.')[0])
            if max_score_frame == -1:
                #just select the 50% of the frames
                max_score_frame = int(len(frame_names) / 2) -1
                fail_count += 1
            cur_key = video + '_' + exp_id
            all_results[cur_key] = max_score_frame

    
    #dump to json
    with open(save_path, 'w') as f:
        json.dump(all_results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")

    args = parser.parse_args()