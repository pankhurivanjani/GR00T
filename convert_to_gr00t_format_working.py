import json
import pandas as pd
import torch
from pathlib import Path
from collections import OrderedDict
import numpy as np
import imageio.v2 as imageio
from PIL import Image
import cv2
from natsort import natsorted

# -----------------------------
# User configuration - adjust these paths as needed
# -----------------------------
SOURCE_DIR = Path("../usb_good_data")  # Path to folder containing .pt files and sensor/top_cam
# OUTPUT_DIR = Path("training_data")    # Path where GR00T-formatted dataset will be created
OUTPUT_DIR = Path("usb_data_lerobot")    # Path where GR00T-formatted dataset will be created
 
# -----------------------------
# Step 1: Create target directory structure
# -----------------------------
# data/chunk-000 for .parquet files
(OUTPUT_DIR / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
# videos/chunk-000/observation.images.ego_view for image frames
#(OUTPUT_DIR / "videos" / "chunk-000" / "observation.images.right_cam").mkdir(parents=True, exist_ok=True)
# meta directory
(OUTPUT_DIR / "meta").mkdir(parents=True, exist_ok=True)

# -----------------------------
# Step 2: Load all .pt tensors
# -----------------------------
def load_tensor(path: Path):
    """Helper to load a torch tensor from a .pt file."""
    return torch.load(path, map_location="cpu",weights_only=True)


def parquet_file_generation(idx, folder, global_idx):
    gripper_state = load_tensor(folder / "202 follower" / "gripper_state.pt")      # shape: [686]
    # print(gripper_state.shape)
    joint_pos = load_tensor(folder / "202 follower" / "joint_pos.pt")       # shape: [686, 7]
    # print(joint_pos.shape)
    # joint_vel = load_tensor(folder / "Panda 10 follower" / "joint_vel.pt")       # shape: [686, 7]
    # print(joint_vel.shape)
    
    # timestamps tensor
    timestamps = load_tensor(folder / "timestamps.pt")                                  # shape: [686]

    task_file = SOURCE_DIR / "task" # earlier (folder/"task")

    with open(task_file, "r") as f:
        #task_idx = int(f.read().strip())
        line = f.readline().strip().split(maxsplit=1)
        task_idx = int(line[0])
        task_description = line[1] if len(line) > 1 else ""
    print(f"task_idx: {task_idx}")

    num_steps = gripper_state.shape[0] - 1

    # Step 3: Write each time step as a single-row Parquet file
    # -----------------------------
    state_parts = OrderedDict([
        ("joint_pos",    joint_pos[:-1,:]),
       #("gripper_state", gripper_state[:-1]),
       ("gripper_state", gripper_state[:-1, None]),
        # ("joint_vel",    joint_vel[:-1,:]),
    ])
    action_parts = OrderedDict([
        ("joint_pos",    joint_pos[1:,:]),
        #("gripper_state", gripper_state[1:]),
        ("gripper_state", gripper_state[1:, None]),
        # ("joint_vel",    joint_vel[1:,:]),
    ])

    rows = []
    for i in range(num_steps): 
        state_vec = np.concatenate([
            arr[i].cpu().numpy().ravel() for arr in state_parts.values()
        ])
        action_vec = np.concatenate([
            arr[i].cpu().numpy().ravel() for arr in action_parts.values()
        ])
        row = {
            "observation.state": state_vec.tolist(),
            "action":            action_vec.tolist(),
            # "timestamp":         float(timestamps[i].item()),
            "timestamp":         0.05 + i * 0.05,
            "annotation.human.action.task_description": task_idx, # index of the task description in the meta/tasks.jsonl file
            "task_index":        task_idx, # index of the task in the meta/tasks.jsonl file
            # "annotation.human.validity": 1, 
            "episode_index":     idx,
            "index":             global_idx,
            "next.reward":       0.0,
            "next.done":         False,
        }
        rows.append(row)
        global_idx += 1

    df = pd.DataFrame(rows)
    df_path = OUTPUT_DIR / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
    df.to_parquet(df_path)
    print(f"episode_{idx:06d}.parquet is generated: {df_path}")

    episodes_jsonl_data = {"episode_index":idx,"tasks":[task_idx],"length":num_steps} # pankhuri fix put task_idx in square brackets to make it a list
    with open(OUTPUT_DIR / "meta"/ "episodes.jsonl", "a") as f:
        f.write(json.dumps(episodes_jsonl_data) + "\n")

    return global_idx


def video_generation(idx: int, folder: Path):
    
    # first camera view
    src = folder / "sensors" / "wrist_cam"
    output = OUTPUT_DIR / "videos" / "chunk-000" / "observation.images.wrist_cam" / f"episode_{idx:06d}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    image_files = natsorted(src.glob("*.*"))
    frames = []
    for img_path in image_files:
        arr = imageio.imread(str(img_path))
        img = Image.fromarray(arr).resize((256, 256), Image.LANCZOS)
        frames.append(np.array(img))

    # ffmpeg_params = [
    #     "-pix_fmt", "yuv420p"      
    #     # "-preset", "fast",            
    #     # "-crf", "23",                
    #     # "-vsync", "passthrough",      
    # ]

    with imageio.get_writer(
        str(output),
        format="FFMPEG",
        mode="I",
        fps=20,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p"]
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"video episode_{idx:06d}.mp4 is generated: {output}")

    # second camera view
    src2 = folder / "sensors" / "right_cam"
    output2 = OUTPUT_DIR / "videos" / "chunk-000" / "observation.images.right_cam" / f"episode_{idx:06d}.mp4"
    output2.parent.mkdir(parents=True, exist_ok=True)

    image_files2 = natsorted(src2.glob("*.*"))
    frames2 = []
    for img_path in image_files2:
        arr = imageio.imread(str(img_path))
        img = Image.fromarray(arr).resize((256, 256), Image.LANCZOS)
        frames2.append(np.array(img))

    with imageio.get_writer(
        str(output2),
        format="FFMPEG",
        mode="I",
        fps=20,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p"]
    ) as writer:
        for frame in frames2:
            writer.append_data(frame)

    print(f"video episode_{idx:06d}.mp4 is generated: {output2}")



# global_idx = 15825 # next global index
# current_epi_idx = 50 # next episode

global_idx = 0 # next global index
current_epi_idx = 0 # next episode

tasks_jsonl = OUTPUT_DIR / "meta" / "tasks.jsonl"
print("SOURCE_DIR:", SOURCE_DIR / "task")
with open(SOURCE_DIR / "task", "r") as f, open(tasks_jsonl, "w") as out_f:
    # debug print the file path and content
    print("Reading task file:", SOURCE_DIR / "task")
    line = f.readline().strip().split(maxsplit=1)
    print("line:", line)
    task_idx = int(line[0])
    task_description = line[1] if len(line) > 1 else ""
    out_f.write(json.dumps({
        "task_index": task_idx,
        "task": task_description
    }) + "\n")


for idx, folder in enumerate(sorted(SOURCE_DIR.iterdir())):    
    if not folder.is_dir():
        continue
    print("*" * 100)
    print(f"{idx+current_epi_idx}: {folder}")
    
    global_idx = parquet_file_generation(idx + current_epi_idx,folder, global_idx)
    
    video_generation(idx + current_epi_idx,folder)

print("-" * 50)
print(f"total numbers: {global_idx }")

