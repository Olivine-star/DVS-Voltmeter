# #! python3
# # -*- encoding: utf-8 -*-
# '''
# @File    :   main.py
# @Time    :   2022/7/12 17:30
# @Author  :   Songnan Lin, Ye Ma
# @Contact :   songnan.lin@ntu.edu.sg, my17@tsinghua.org.cn
# @Note    :   
# @inproceedings{lin2022dvsvoltmeter,
#   title={DVS-Voltmeter: Stochastic Process-based Event Simulator for Dynamic Vision Sensors},
#   author={Lin, Songnan and Ma, Ye and Guo, Zhenhua and Wen, Bihan},
#   booktitle={ECCV},
#   year={2022}
# }
# '''

# import argparse
# import os
# import numpy as np
# import cv2
# import tqdm
# from src.config import cfg
# from src.simulator import EventSim
# from src.visualize import events_to_voxel_grid, visual_voxel_grid

# def get_args_from_command_line():
#     parser = argparse.ArgumentParser(description='Parser of Runner of Network')
#     parser.add_argument('--camera_type', type=str, help='Camera type, such as DVS346', default='DVS346')
#     parser.add_argument('--model_para', type=float, nargs='+', help='Set parameters for a specific camera type', default=None)
#     parser.add_argument('--input_dir', type=str, help='Set dataset root_path', default=None)
#     parser.add_argument('--output_dir', type=str, help='Set output path', default=None)
#     args = parser.parse_args()
#     return args

# def integrate_cfg(cfg, command_line_args):
#     args = command_line_args
#     cfg.SENSOR.CAMERA_TYPE = args.camera_type if args.camera_type is not None else cfg.SENSOR.CAMERA_TYPE
#     cfg.SENSOR.K = args.model_para if args.model_para is not None else cfg.SENSOR.K
#     cfg.DIR.IN_PATH = args.input_dir if args.input_dir is not None else cfg.DIR.IN_PATH
#     cfg.DIR.OUT_PATH = args.output_dir if args.output_dir is not None else cfg.DIR.OUT_PATH
#     if cfg.SENSOR.K is None or len(cfg.SENSOR.K) != 6:
#         raise Exception('No model parameters given for sensor type %s' % cfg.SENSOR.CAMERA_TYPE)
#     print(cfg)
#     return cfg

# def is_valid_dir(dirs):
#     return os.path.exists(os.path.join(dirs, 'info.txt'))

# def process_dir(cfg, file_info, video_name):
#     indir = os.path.join(cfg.DIR.IN_PATH, video_name)
#     outdir = os.path.join(cfg.DIR.OUT_PATH, video_name)
#     print(f"Processing folder {indir}... Generating events in file {outdir}")

#     # file info
#     file_timestamps_us = [int(info_i.split()[1]) for info_i in file_info]
#     file_paths = [info_i.split()[0] for info_i in file_info]

#     # set simulator
#     sim = EventSim(cfg=cfg, output_folder=cfg.DIR.OUT_PATH, video_name=video_name)

#     # process
#     pbar = tqdm.tqdm(total=len(file_paths))
#     num_events, num_on_events, num_off_events = 0, 0, 0
#     events = []
#     for i in range(0, len(file_paths)):
#         timestamp_us = file_timestamps_us[i]
#         image = cv2.imread(file_paths[i], cv2.IMREAD_GRAYSCALE)

#         # event generation!!!
#         event = sim.generate_events(image, timestamp_us)

#         if event is not None:
#             events.append(event)
#             num_events += event.shape[0]
#             num_on_events += np.sum(event[:, -1] == 1)
#             num_off_events += np.sum(event[:, -1] == 0)

#         pbar.set_description(f"Events generated: {num_events}")
#         pbar.update(1)

#     events = np.concatenate(events, axis=0)
#     np.savetxt(os.path.join(cfg.DIR.OUT_PATH, video_name + '.txt'), events, fmt='%1.0f')
#     sim.reset()


# if __name__ == "__main__":
#     args = get_args_from_command_line()
#     cfg = integrate_cfg(cfg, args)

#     video_list = sorted(os.listdir(cfg.DIR.IN_PATH))
#     for video_i in video_list:
#         video_i_path = os.path.join(cfg.DIR.IN_PATH, video_i)
#         os.makedirs(os.path.join(cfg.DIR.OUT_PATH, video_i), exist_ok=True)

#         if is_valid_dir(video_i_path):
#             # video info
#             with open(os.path.join(cfg.DIR.IN_PATH, video_i, 'info.txt'), 'r') as f:
#                 video_info = f.readlines()
#             # simulation
#             process_dir(cfg=cfg, file_info=video_info, video_name=video_i)





import argparse
import os
import re
import numpy as np
import cv2
import tqdm
from src.config import cfg
from src.simulator import EventSim
from src.visualize import events_to_voxel_grid, visual_voxel_grid

def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='Parser of Runner of Network')
    parser.add_argument('--camera_type', type=str, default='DVS346')
    parser.add_argument('--model_para', type=float, nargs='+', default=None)
    parser.add_argument('--input_dir', type=str, default=None)   # e.g. processed root
    parser.add_argument('--output_dir', type=str, default=None)  # e.g. event_process root
    args = parser.parse_args()
    return args

def integrate_cfg(cfg, command_line_args):
    args = command_line_args
    cfg.SENSOR.CAMERA_TYPE = args.camera_type if args.camera_type is not None else cfg.SENSOR.CAMERA_TYPE
    cfg.SENSOR.K = args.model_para if args.model_para is not None else cfg.SENSOR.K
    cfg.DIR.IN_PATH = args.input_dir if args.input_dir is not None else cfg.DIR.IN_PATH
    cfg.DIR.OUT_PATH = args.output_dir if args.output_dir is not None else cfg.DIR.OUT_PATH
    if cfg.SENSOR.K is None or len(cfg.SENSOR.K) != 6:
        raise Exception('No model parameters given for sensor type %s' % cfg.SENSOR.CAMERA_TYPE)
    print(cfg)
    return cfg

def is_valid_dir(d):
    return os.path.exists(os.path.join(d, 'info.txt'))

# 新增：递归查找所有包含 info.txt 的目录，并解析 seq 与 scale
def find_video_dirs(root):
    """
    返回列表：[(dir_with_info, seq_name, scale_factor_str), ...]
    其中 scale_factor_str 是 '2' / '4' / '8'
    """
    results = []
    for cur, dirs, files in os.walk(root):
        if 'info.txt' in files:
            # 解析 seq（向上找 Indoor*/Outdoor*）
            parts = cur.replace('\\','/').split('/')
            seq_name = None
            for p in reversed(parts):
                if re.match(r'(?i)indoor\d+|outdoor\d+', p):
                    seq_name = p
                    break
            if seq_name is None:
                seq_name = os.path.basename(root)

            # 解析 scale（路径中含 ds_x2 / ds_x4 / ds_x8）
            m = re.search(r'ds_x(\d+)', cur.replace('\\','/'), re.IGNORECASE)
            scale = m.group(1) if m else None  # '2'/'4'/'8'

            results.append((cur, seq_name, scale))
    return results

# 修改：保存为 OUT_PATH/down{2,4,8}/{seq}.txt，并写 header 兼容 skiprows=1
def process_dir_and_save(cfg, file_info, seq_name, scale):
    # scale->子目录映射
    scale_map = {'2': 'down2', '4': 'down4', '8': 'down8'}
    if scale not in scale_map:
        # 若没解析到 scale，就放到 down2（或给出提示）
        print(f"[WARN] Scale not detected for seq={seq_name}. Defaulting to down2.")
        out_sub = 'down2'
    else:
        out_sub = scale_map[scale]

    out_dir = os.path.join(cfg.DIR.OUT_PATH, out_sub)
    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, f"{seq_name}.txt")

    # 读取 info.txt
    file_timestamps_us = [int(line.split()[1]) for line in file_info]
    file_paths = [line.split()[0] for line in file_info]

    # set simulator
    sim = EventSim(cfg=cfg, output_folder=cfg.DIR.OUT_PATH, video_name=f"{seq_name}_{out_sub}")

    # process
    pbar = tqdm.tqdm(total=len(file_paths))
    num_events = 0
    events = []
    for i in range(len(file_paths)):
        timestamp_us = file_timestamps_us[i]
        image = cv2.imread(file_paths[i], cv2.IMREAD_GRAYSCALE)
        event = sim.generate_events(image, timestamp_us)
        if event is not None:
            events.append(event)
            num_events += event.shape[0]
        pbar.set_description(f"{seq_name} {out_sub} | Events: {num_events}")
        pbar.update(1)

    if len(events) == 0:
        print(f"[WARN] No events generated for {seq_name} {out_sub}")
        sim.reset()
        return

    events = np.concatenate(events, axis=0)

    # 关键：写 header，方便你下游 pandas skiprows=1
    np.savetxt(out_txt, events, fmt='%1.0f', header='t x y pol', comments='')
    sim.reset()
    print(f"[OK] {out_txt}  ({events.shape[0]} events)")

if __name__ == "__main__":
    args = get_args_from_command_line()
    cfg = integrate_cfg(cfg, args)

    # 递归发现所有包含 info.txt 的目录
    video_dirs = find_video_dirs(cfg.DIR.IN_PATH)
    if not video_dirs:
        raise SystemExit(f"No dirs with info.txt found under {cfg.DIR.IN_PATH}")

    print(f"Found {len(video_dirs)} dirs with info.txt")
    for d, seq, scale in sorted(video_dirs):
        with open(os.path.join(d, 'info.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        process_dir_and_save(cfg=cfg, file_info=lines, seq_name=seq, scale=scale)
