"""Save sleeping/lying/fall annotated frames from video105."""
import cv2
from pathlib import Path

bbox_dir = Path('dataset/video105/BBOX')
rgb_dir  = Path('dataset/video105/RGB')
out_dir  = Path('output_results/frames/video105_sleeping')
out_dir.mkdir(parents=True, exist_ok=True)

action_map = {0:'sit',1:'stand',2:'walk',3:'fall',4:'lie',5:'standup',6:'sitdown',7:'lying',8:'sleeping',9:'falling',10:'falled',11:'active'}
colors     = {7:(255,100,230),8:(30,100,255),9:(0,180,255),10:(50,50,180),4:(200,80,255)}
TARGET_CLS = {4, 7, 8, 9, 10}

def get_fnum(path):
    stem = Path(path).stem
    idx  = stem.rfind('_frame_')
    return int(stem[idx+7:]) if idx != -1 else -1

rgb_map = {get_fnum(p): p for p in rgb_dir.glob('*.jpg')}
saved   = 0

for bbox_file in sorted(bbox_dir.glob('*.txt'), key=lambda p: get_fnum(p.name)):
    if bbox_file.stat().st_size == 0:
        continue
    with open(bbox_file) as f:
        lines = [l.strip().split() for l in f if l.strip()]
    if not lines:
        continue
    classes = [int(l[0]) for l in lines if len(l) >= 5]
    if not any(c in TARGET_CLS for c in classes):
        continue
    fnum = get_fnum(bbox_file.name)
    if fnum not in rgb_map:
        continue
    frame = cv2.imread(str(rgb_map[fnum]))
    if frame is None:
        continue
    for l in lines:
        if len(l) < 5:
            continue
        cls  = int(l[0])
        cx, cy, w, h = map(float, l[1:5])
        x1 = int((cx - w/2) * 320); y1 = int((cy - h/2) * 240)
        x2 = int((cx + w/2) * 320); y2 = int((cy + h/2) * 240)
        color = colors.get(cls, (200,200,200))
        label = action_map.get(cls, str(cls))
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
        cv2.putText(frame, label, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20,20,20), 1, cv2.LINE_AA)
    label_tag = '_'.join(set(action_map.get(c, str(c)) for c in classes if c in TARGET_CLS))
    out_path  = out_dir / f'frame_{fnum:05d}_{label_tag}.jpg'
    cv2.imwrite(str(out_path), frame)
    saved += 1
    if saved >= 8:
        break

print(f'Saved {saved} frames to: {out_dir}')
for f in sorted(out_dir.glob('*.jpg')):
    print(' ', f.name)
