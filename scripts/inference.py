import os
import torch
import gr00t

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy
import torch

import sys

# REPO_PATH is the path of the pip install gr00t repo and one level up
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATASET_PATH = os.path.join(REPO_PATH, "training_data")
MODEL_PATH = os.path.join(REPO_PATH, "checkpoints/checkpoint_20250622_093449")
EMBODIMENT_TAG = "new_embodiment"

device = "cuda" if torch.cuda.is_available() else "cpu"


# Loading Pretrained Policy

from gr00t.experiment.data_config import DATA_CONFIG_MAP


data_config = DATA_CONFIG_MAP["irl_panda"]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag=EMBODIMENT_TAG,
    modality_config=modality_config,
    modality_transform=modality_transform,
    device=device,
)

# print out the policy model architecture
# print(policy.model)


#Loading dataset
import numpy as np

modality_config = policy.modality_config

print(modality_config.keys())

for key, value in modality_config.items():
    if isinstance(value, np.ndarray):
        print(key, value.shape)
    else:
        print(key, value)

# Create the dataset
dataset = LeRobotSingleDataset(
    dataset_path=DATASET_PATH,
    modality_configs=modality_config,
    video_backend="decord",
    video_backend_kwargs=None,
    transforms=None,  # We'll handle transforms separately through the policy
    embodiment_tag=EMBODIMENT_TAG,
)

import numpy as np

step_data = dataset[300]

print(step_data)

print("\n\n ====================================")
for key, value in step_data.items():
    if isinstance(value, np.ndarray):
        print(key, value.shape)
    else:
        print(key, value)


import matplotlib.pyplot as plt

traj_id = 15
max_steps = 300

state_joints_across_time = []
gt_action_joints_across_time = []

images = []

sample_images = 6

for step_count in range(max_steps):
    data_point = dataset.get_step_data(traj_id, step_count)
    state_joints = data_point["state.joint_pos"][0]
    gt_action_joints = data_point["action.joint_pos"][0]
    state_joints_across_time.append(state_joints)
    gt_action_joints_across_time.append(gt_action_joints)

    # We can also get the image data
    if step_count % (max_steps // sample_images) == 0:
        image = data_point["video.ego_view"][0]
        images.append(image)

# Size is (max_steps, num_joints == 7)
state_joints_across_time = np.array(state_joints_across_time)
gt_action_joints_across_time = np.array(gt_action_joints_across_time)


# Plot the joint angles across time
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(8, 2*7))

for i, ax in enumerate(axes):
    ax.plot(state_joints_across_time[:, i], label="state joints")
    ax.plot(gt_action_joints_across_time[:, i], label="gt action joints")
    ax.set_title(f"Joint {i}")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, "output_action.png"))
print("figure is saved 😀")   
# plt.show()




state_gripper_across_time = []
gt_action_gripper_across_time = []

for step_count in range(max_steps):
    data_point = dataset.get_step_data(traj_id, step_count)
    state_gripper = data_point["state.gripper_state"][0]  # shape: (1,)
    gt_gripper = data_point["action.gripper_state"][0]    # shape: (1,)

    state_gripper_across_time.append(state_gripper)
    gt_action_gripper_across_time.append(gt_gripper)

# 转换为 NumPy array
state_gripper_across_time = np.array(state_gripper_across_time)  # shape: (max_steps, 1)
gt_action_gripper_across_time = np.array(gt_action_gripper_across_time)

# 绘图
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(state_gripper_across_time[:, 0], label="state gripper")
ax.plot(gt_action_gripper_across_time[:, 0], label="gt action gripper")
ax.set_title("Gripper")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, "output_gripper.png"))
print("figure is saved 😀")





# Plot the images in a row
fig, axes = plt.subplots(nrows=1, ncols=sample_images, figsize=(16, 4))

for i, ax in enumerate(axes):
    ax.imshow(images[i])
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, "output_video.png"))
print("figure is saved 😀")   


predicted_action = policy.get_action(step_data)
for key, value in predicted_action.items():
    print(key, value.shape)


# --------------------------------------------
# Step-wise prediction vs GT: joint positions
# --------------------------------------------
pred_joint_seq = []
gt_joint_seq = []

for step in range(max_steps):
    step_data = dataset.get_step_data(traj_id, step)
    
    # 模型预测动作
    pred_action = policy.get_action(step_data)
    pred_joint = pred_action["action.joint_pos"][0]  # shape: (7,)
    gt_joint = step_data["action.joint_pos"][0]      # shape: (7,)

    pred_joint_seq.append(pred_joint)
    gt_joint_seq.append(gt_joint)

# 转换为 NumPy 数组: (steps, joints)
pred_joint_seq = np.array(pred_joint_seq)
gt_joint_seq = np.array(gt_joint_seq)

# 绘图
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(10, 2 * 7))

for i, ax in enumerate(axes):
    ax.plot(gt_joint_seq[:, i], label="GT", color="green")
    ax.plot(pred_joint_seq[:, i], label="Pred", color="red", linestyle="--")
    ax.set_title(f"Joint {i}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Joint Pos (rad)")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, "output_pred_vs_gt_joint_sequence.png"))
print("✅ Step-wise joint prediction vs GT figure saved!")


# --------------------------------------------
# Step-wise prediction vs GT: gripper state
# --------------------------------------------
pred_gripper_seq = []
gt_gripper_seq = []

for step in range(max_steps):
    step_data = dataset.get_step_data(traj_id, step)

    pred_action = policy.get_action(step_data)
    pred_gripper = pred_action["action.gripper_state"][0]  # scalar
    gt_gripper = step_data["action.gripper_state"][0]      # scalar

    pred_gripper_seq.append(pred_gripper)
    gt_gripper_seq.append(gt_gripper)

# 转换为 NumPy array
pred_gripper_seq = np.array(pred_gripper_seq)  # shape: (steps,)
gt_gripper_seq = np.array(gt_gripper_seq)

# 绘图
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(gt_gripper_seq, label="GT", color="green")
ax.plot(pred_gripper_seq, label="Pred", color="red", linestyle="--")
ax.set_title("Gripper State Over Time")
ax.set_xlabel("Step")
ax.set_ylabel("Gripper Command")
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_PATH, "output_pred_vs_gt_gripper_sequence.png"))
print("✅ Step-wise gripper prediction vs GT figure saved!")

