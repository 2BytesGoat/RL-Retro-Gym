import cv2
from pathlib import Path

video_path = Path('./data/raw/openaigym.video.0.6676.video000000.mp4')
max_frames = 2000

dump_path = video_path.parents[1] / 'splitted'
dump_name = video_path.stem + '_{}.png'

dump_path.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(video_path))

frame_nb = 0
while cap.isOpened() and frame_nb < max_frames:
    ret, frame = cap.read()
    frame_path = str(dump_path / dump_name.format(frame_nb))
    cv2.imwrite(frame_path, frame)
    frame_nb += 1

cap.release()
