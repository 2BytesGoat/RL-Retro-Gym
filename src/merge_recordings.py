import os
import json
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from processing.preprocessing import fetch_region_pipeline
from utils import load_config

def get_csv_path_from_video_path(video_path):
    rec_index = video_path.stem[-1]
    csv_path = f'{video_path.parent}/openaigym.episode_summary.{rec_index}.csv'
    return csv_path

def get_video_frames(video_path, config):
    pre_pipeline = fetch_region_pipeline(**config['preprocessing'])

    frames = []
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        formatted_frame = pre_pipeline.fit_transform(frame)
        frames.append(formatted_frame.flatten() / 255.0)
    cap.release()

    return frames

def framest_to_df(frames):
    nb_pixels = len(frames[0])
    frames_df = pd.DataFrame(frames, columns=[f'pixel{i}' for i in range(nb_pixels)])
    return frames_df

def extract_actions(meta_df):
    unique_actions = sorted(set(meta_df['action']))
    action_mapping = {
        action: onehot for action, onehot in zip(unique_actions, range(len(unique_actions)))
    }
    target = meta_df['action'].map(action_mapping)
    return target, action_mapping

if __name__ == '__main__':
    src_data_path = 'data/gameplay_rec'
    dst_data_path = 'data/mnist_appoch'
    config_path = 'configs/mnist_state_trasnform.json'

    os.makedirs(dst_data_path, exist_ok=True)
    config = load_config(config_path)

    all_dfs = []
    for video_path in tqdm(Path(src_data_path).glob('*/*.mp4')):
        csv_path = get_csv_path_from_video_path(video_path)

        frames = get_video_frames(video_path, config)
        meta_df = pd.read_csv(csv_path)

        frames_df = framest_to_df(frames)
        frames_df['action'] = meta_df['action']
        frames_df = frames_df.dropna()

        mask = frames_df['action'].apply(lambda x: x in config['limited_actions'])
        frames_df = frames_df[mask]

        all_dfs.append(frames_df)

    merged_df = pd.concat(all_dfs)

    target, action_mapping = extract_actions(merged_df)
    merged_df['label'] = target
    merged_df.to_csv(f'{dst_data_path}/train_df.csv', index=False)

    output_mapping = {value: key for key, value in action_mapping.items()}
    json.dump(action_mapping, open(f'{dst_data_path}/action_mapping.json', 'w'))
    json.dump(output_mapping, open(f'{dst_data_path}/output_mapping.json', 'w'))
