import cv2
import random
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=Path)
    parser.add_argument("--txt_path", required=True, type=Path)
    parser.add_argument("--max_frames", default=2000)
    parser.add_argument("--frame_stride", default=1, type=int)
    parser.add_argument("--paths_split", default=0.75)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.data_path.is_file():
        dump_folder_name = args.data_path.parents[0].stem + '_splitted'
        dump_path =  args.data_path.parents[1] / dump_folder_name
        extract_from_file(args.data_path, dump_path, args.max_frames, args, append_txt=False)

    elif args.data_path.is_dir():
        dump_folder_name = args.data_path.stem + '_splitted'
        dump_path =  args.data_path.parents[0] / dump_folder_name
        video_paths = list(args.data_path.glob('*.mp4'))
        max_frames = int(args.max_frames / len(video_paths))
        for video in video_paths:
            extract_from_file(video, dump_path, max_frames, args, append_txt=True)


def extract_from_file(file_path, dump_path, max_frames, args, append_txt):
    file_paths = dump_frames(file_path, dump_path, max_frames, args.frame_stride)

    train_paths = random.choices(file_paths, k=int(len(file_paths) * args.paths_split))
    valid_paths = list(set(file_paths) - set(train_paths))
    export_paths(train_paths, args.txt_path / 'train_files.txt', append_txt)
    export_paths(valid_paths, args.txt_path / 'valid_files.txt', append_txt)


def dump_frames(file_path, dump_path, max_frames, frame_stride):
    dump_name = file_path.stem + '_{}.png'
    dump_path.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(file_path))

    frame_nb = 0
    file_paths = []
    while cap.isOpened() and len(file_paths) < max_frames:
        ret, frame = cap.read()
        if frame is None:
            break
        frame_nb += 1
        if frame_nb % frame_stride != 0:
            continue
        frame_path = str(dump_path / dump_name.format(frame_nb))
        cv2.imwrite(frame_path, frame)
        file_paths.append(frame_path)

    cap.release()
    return file_paths


def export_paths(paths, dst_file, append_txt):
    mod = 'a' if append_txt else 'w'
    with open(dst_file, mode=mod) as dest_f:
        for path in paths:
            dest_f.write(str(path) + '\n')

if __name__ == '__main__':
    main()