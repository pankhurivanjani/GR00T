import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
import gr00t


# -----------------------------
# Configurations
# -----------------------------
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATASET_PATH = os.path.join(REPO_PATH, "usb_data_lerobot")   # "training_data"
MODEL_PATH = os.path.join(REPO_PATH, "checkpoints/multi_task_0_120250925_191136/checkpoint-60000") ##checkpoint_20250622_093449")
EMBODIMENT_TAG = "new_embodiment"
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Initialization
# -----------------------------
def load_policy():
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
    return policy, modality_config

def load_dataset(modality_config):
    return LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=EMBODIMENT_TAG,
    )


# -----------------------------
# Visualization Functions
# -----------------------------
def plot_joint_states(dataset, traj_id=0, max_steps=300, sample_images=6):
    state_seq, action_seq, images = [], [], []

    for step in range(max_steps):
        data = dataset.get_step_data(traj_id, step)
        state_seq.append(data["state.joint_pos"][0])
        action_seq.append(data["action.joint_pos"][0])
        if step % (max_steps // sample_images) == 0:
            images.append(data["video.right_cam"][0])
            images.append(data["video.wrist_cam"][0])
            #images.append(data["video.ego_view"][0])

    state_seq = np.array(state_seq)
    action_seq = np.array(action_seq)

    fig, axes = plt.subplots(7, 1, figsize=(8, 14))
    for i in range(7):
        axes[i].plot(state_seq[:, i], label="state")
        axes[i].plot(action_seq[:, i], label="action")
        axes[i].set_title(f"Joint {i}")
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, "output_action.png"))

    # plot images
    fig, axes = plt.subplots(1, sample_images, figsize=(16, 4))
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, "output_video.png"))


def plot_gripper_states(dataset, traj_id=0, max_steps=300):
    state_seq, action_seq = [], []
    for step in range(max_steps):
        data = dataset.get_step_data(traj_id, step)
        state_seq.append(data["state.gripper_state"][0])
        action_seq.append(data["action.gripper_state"][0])

    state_seq = np.array(state_seq)
    action_seq = np.array(action_seq)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(state_seq, label="state gripper")
    ax.plot(action_seq, label="gt gripper")
    ax.set_title("Gripper State")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, "output_gripper.png"))


# -----------------------------
# Prediction vs GT Comparison
# -----------------------------
def compare_joint_predictions(policy, dataset, traj_id=0, max_steps=300):
    pred_seq, gt_seq = [], []

    for step in range(max_steps):
        data = dataset.get_step_data(traj_id, step)
        pred = policy.get_action(data)["action.joint_pos"][0]
        gt = data["action.joint_pos"][0]
        pred_seq.append(pred)
        gt_seq.append(gt)

    pred_seq = np.array(pred_seq)
    gt_seq = np.array(gt_seq)

    fig, axes = plt.subplots(7, 1, figsize=(10, 14))
    for i in range(7):
        axes[i].plot(gt_seq[:, i], label="GT", color="green")
        axes[i].plot(pred_seq[:, i], label="Pred", linestyle="--", color="red")
        axes[i].set_title(f"Joint {i}")
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, "output_pred_vs_gt_joint_sequence.png"))

def compare_gripper_predictions(policy, dataset, traj_id=0, max_steps=300):
    pred_seq, gt_seq = [], []

    for step in range(max_steps):
        data = dataset.get_step_data(traj_id, step)
        pred = policy.get_action(data)["action.gripper_state"][0]
        gt = data["action.gripper_state"][0]
        pred_seq.append(pred)
        gt_seq.append(gt)

    pred_seq = np.array(pred_seq)
    gt_seq = np.array(gt_seq)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gt_seq, label="GT", color="green")
    ax.plot(pred_seq, label="Pred", linestyle="--", color="red")
    ax.set_title("Gripper Prediction")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, "output_pred_vs_gt_gripper_sequence.png"))


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    policy, modality_config = load_policy()
    dataset = load_dataset(modality_config)

    traj_id = 0
    max_steps = 300

    print("🔧 Running GT visualization...")
    plot_joint_states(dataset, traj_id, max_steps)
    plot_gripper_states(dataset, traj_id, max_steps)

    print("🤖 Running predictions...")
    compare_joint_predictions(policy, dataset, traj_id, max_steps)
    compare_gripper_predictions(policy, dataset, traj_id, max_steps)

    print("✅ All visualizations saved in:", MODEL_PATH)
