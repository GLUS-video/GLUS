import json
import os

joint_path = 'PATH_JOINT_SAVE'

mevis_path = 'PATH_TO_MEVIS_INIT_INFERENCE'

youtube_path = 'PATH_TO_YOUTUBE_INFERENCE'

mevis_folders = os.listdir(mevis_path)
youtube_folders = os.listdir(youtube_path)
revos_folders = os.listdir(revos_path)

#copy all folders in mevis and train into joint
os.makedirs(joint_path, exist_ok=True)
for folder in mevis_folders:
    os.system(f'cp -r {os.path.join(mevis_path, folder)} {os.path.join(joint_path, folder)}')
for folder in youtube_folders:
    os.system(f'cp -r {os.path.join(youtube_path, folder)} {os.path.join(joint_path, folder)}')
for folder in revos_folders:
    os.system(f'cp -r {os.path.join(revos_path, folder)} {os.path.join(joint_path, folder)}')
