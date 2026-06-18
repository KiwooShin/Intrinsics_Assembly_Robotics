"""Render a side-by-side (L/C/R) video of the CheatCode SFP insertion from a bag.
Trims to the task window (first..last pose_command) and encodes h264 via ffmpeg."""
import sys, os, subprocess
sys.path.insert(0, '/home/kiwoos/miniconda3/lib/python3.13/site-packages')
import numpy as np, cv2
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

bag = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/data/demos/one_20260617_233031')
out = sys.argv[2] if len(sys.argv) > 2 else '/home/kiwoos/work/insertion_demo.mp4'
FPS = 12
CW, CH = 432, 384          # per-camera (aspect 1152/1024 = 1.125)
HEADER = 40
ts = get_typestore(Stores.ROS2_KILTED)

imgs = {'left': [], 'center': [], 'right': []}
cmd_times = []
print(f"Reading {bag} ...")
with Reader(bag) as reader:
    for conn, t_ns, raw in reader.messages():
        t = t_ns * 1e-9
        if conn.topic in ('/left_camera/image', '/center_camera/image', '/right_camera/image'):
            cam = conn.topic.split('/')[1].replace('_camera', '')
            m = ts.deserialize_cdr(raw, conn.msgtype)
            a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            imgs[cam].append((t, a))
        elif conn.topic == '/aic_controller/pose_commands':
            cmd_times.append(t)

t0, t1 = min(cmd_times), max(cmd_times) + 0.3
center = [(t, a) for (t, a) in imgs['center'] if t0 <= t <= t1]
print(f"task window {t1-t0:.1f}s | center frames in window: {len(center)}")

W = CW * 3
H = CH + HEADER
_fflog = open('/tmp/ffmpeg_video.log', 'wb')
ff = subprocess.Popen(
    ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
     '-r', str(FPS), '-i', '-', '-c:v', 'libopenh264', '-pix_fmt', 'yuv420p',
     '-b:v', '5M', out],
    stdin=subprocess.PIPE, stdout=_fflog, stderr=_fflog)

def nearest(lst, t):
    return min(lst, key=lambda x: abs(x[0] - t))[1]

N = len(center)
for i, (t, c) in enumerate(center):
    l = nearest(imgs['left'], t); r = nearest(imgs['right'], t)
    tiles = []
    for name, im in (('LEFT', l), ('CENTER', c), ('RIGHT', r)):
        im = cv2.cvtColor(cv2.resize(im, (CW, CH)), cv2.COLOR_RGB2BGR)
        cv2.putText(im, name, (10, CH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        tiles.append(im)
    row = np.hstack(tiles)
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[HEADER:] = row
    cv2.putText(canvas, "AIC SFP insertion  -  CheatCode (score 93.2)", (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, f"t={t-t0:5.1f}s  {i+1}/{N}", (W - 230, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    # progress bar
    cv2.rectangle(canvas, (10, 33), (W - 240, 37), (60, 60, 60), -1)
    cv2.rectangle(canvas, (10, 33), (10 + int((W - 250) * (i + 1) / N), 37), (0, 220, 255), -1)
    ff.stdin.write(canvas.tobytes())

ff.stdin.close(); ff.wait()
sz = os.path.getsize(out) / 1e6
print(f"Wrote {out}  ({N} frames @ {FPS}fps = {N/FPS:.1f}s, {sz:.1f} MB)")
