from pathlib import Path

SOURCE_DIR = Path("/hkfs/work/workspace/scratch/vb0283-fastslowtac/usb_good_data") 

task_id = 0
count = 0
for folder in sorted(SOURCE_DIR.iterdir()):

    count +=1

print(f"total episode numuber in the data folder is : {count}")

#count current total length

import json

total_length = 0
file_path = "/hkfs/work/workspace/scratch/vb0283-fastslowtac/GR00T/usb_data_lerobot/meta/episodes.jsonl"

with open(file_path, 'r') as file:
    for line in file:
        if line.strip():
            data = json.loads(line)
            total_length += data["length"]

print(f"Total length: {total_length}")