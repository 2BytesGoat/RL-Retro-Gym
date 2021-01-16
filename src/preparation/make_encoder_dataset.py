import cv2
import random
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, type=Path)
    parser.add_argument("--max_frames", default=2000)
    parser.add_argument("--skip_frames", default=0, type=int)
    parser.add_argument("--export_paths", action='store_true')
    parser.add_argument("--paths_split", default=0.75)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    file_paths = dump_frames(args.video_path, args.max_frames, args.skip_frames)
    if args.export_paths:
        train_paths = random.choices(file_paths, k=int(len(file_paths) * args.paths_split))
        valid_paths = list(set(file_paths) - set(train_paths))
        export_paths(train_paths, args.video_path.parents[1] / 'train_files.txt')
        export_paths(valid_paths, args.video_path.parents[1] / 'valid_files.txt')

def dump_frames(video_path, max_frames, skip_frames):
    dump_path = video_path.parents[1] / 'splitted'
    dump_name = video_path.stem + '_{}.png'
    dump_path.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    frame_nb = 0
    file_paths = []
    while cap.isOpened() and len(file_paths) < max_frames:
        ret, frame = cap.read()
        frame_nb += 1
        if frame_nb <= skip_frames:
            continue
        frame_path = str(dump_path / dump_name.format(frame_nb))
        cv2.imwrite(frame_path, frame)
        file_paths.append(frame_path)

    cap.release()
    return file_paths

def export_paths(paths, dst_file):
    with open(dst_file, mode='w') as dest_f:
        for path in paths:
            dest_f.write(str(path) + '\n')

if __name__ == '__main__':
    main()