import random
import numpy as np

import time
import torch
from torch.utils.data import Dataset
from pynput import keyboard
from functools import partial
from collections import deque



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class KeyboardObserver:
    def __init__(self):
        self.reset()
        self.hotkeys = keyboard.GlobalHotKeys(
            {
                "g": partial(self.set_label, 1),  # good
                "b": partial(self.set_label, 0),  # bad
                "x": self.reset_episode,
            }
        )
        self.hotkeys.start()
        self.direction = np.array([0.0,0.0])
        self.listener = keyboard.Listener(
            on_press=self.set_direction, on_release=self.reset_direction
        )
        self.key_mapping = {
            "a": (1, [1.5,3.0]),  # left
            "d": (1, [3.0,1.5]),  # right
            "w": (0, [3.0,3.0]),  # backward
        }
        self.listener.start()
        return

    def set_label(self, value):
        self.label = value
        print("label set to: ", value)
        return

    def get_label(self):
        return self.label


    def set_direction(self, key):
        try:
            idx, value = self.key_mapping[key.char]
            self.direction = value
        except (KeyError, AttributeError):
            pass
        return

    def reset_direction(self, key):
        try:
            idx, _ = self.key_mapping[key.char]
            self.direction = [0.0,0.0]
        except (KeyError, AttributeError):
            pass
        return

    def has_joints_cor(self):
        return self.direction[0]>0.1

    def reset_episode(self):
        self.reset_button = True
        return

    def reset(self):
        self.set_label(1)
        self.reset_button = False
        return





class TrajectoriesDataset(Dataset):
    def __init__(self, sequence_len):
        self.sequence_len = sequence_len
        self.camera_obs = []
        self.proprio_obs = []
        self.action = []
        self.feedback = []
        self.reset_current_traj()
        self.pos_count = 0
        self.cor_count = 0
        self.neg_count = 0
        return

    def __getitem__(self, idx):
        if self.cor_count < 10:
            alpha = 1
        else:
            alpha = (self.pos_count + self.neg_count) / self.cor_count
        weighted_feedback = [
            alpha if value == -1 else value for value in self.feedback[idx]
        ]
        weighted_feedback = torch.tensor(weighted_feedback).unsqueeze(1)
        return (
            self.camera_obs[idx],
            self.proprio_obs[idx],
            self.action[idx],
            weighted_feedback,
        )

    def __len__(self):
        return len(self.proprio_obs)

    def add(self, camera_obs, proprio_obs, action, feedback):
        self.current_camera_obs.append(camera_obs)
        self.current_proprio_obs.append(proprio_obs)
        self.current_action.append(action)
        self.current_feedback.append(feedback)
        if feedback[0] == 1:
            self.pos_count += 1
        elif feedback[0] == -1:
            self.cor_count += 1
        elif feedback[0] == 0:
            self.neg_count += 1
        return

    def save_current_traj(self):
        camera_obs = downsample_traj(self.current_camera_obs, self.sequence_len)
        proprio_obs = downsample_traj(self.current_proprio_obs, self.sequence_len)
        action = downsample_traj(self.current_action, self.sequence_len)
        feedback = downsample_traj(self.current_feedback, self.sequence_len)
        camera_obs_th = torch.tensor(camera_obs, dtype=torch.float32)
        proprio_obs_th = torch.tensor(proprio_obs, dtype=torch.float32)
        action_th = torch.tensor(action, dtype=torch.float32)
        feedback_th = torch.tensor(feedback, dtype=torch.float32)
        self.camera_obs.append(camera_obs_th)
        self.proprio_obs.append(proprio_obs_th)
        self.action.append(action_th)
        self.feedback.append(feedback_th)
        self.reset_current_traj()
        return

    def reset_current_traj(self):
        self.current_camera_obs = []
        self.current_proprio_obs = []
        self.current_action = []
        self.current_feedback = []
        return

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self))
        indeces = random.sample(range(len(self)), batch_size)
        batch = zip(*[self[i] for i in indeces])
        camera_batch = torch.stack(next(batch), dim=1)
        proprio_batch = torch.stack(next(batch), dim=1)
        action_batch = torch.stack(next(batch), dim=1)
        feedback_batch = torch.stack(next(batch), dim=1)
        return camera_batch, proprio_batch, action_batch, feedback_batch


def downsample_traj(traj, target_len):
    if len(traj) == target_len:
        return traj
    elif len(traj) < target_len:
        return traj + [traj[-1]] * (target_len - len(traj))
    else:
        indeces = np.linspace(start=0, stop=len(traj) - 1, num=target_len)
        indeces = np.round(indeces).astype(int)
        return np.array([traj[i] for i in indeces])


def loop_sleep(start_time):
    dt = 0.05
    sleep_time = dt - (time.time() - start_time)
    if sleep_time > 0.0:
        time.sleep(sleep_time)
    return



def set_seeds(seed=0):
    """Sets all seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
