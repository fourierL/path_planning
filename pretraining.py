import torch
from models import Policy
from utils import TrajectoriesDataset  # noqa: F401
from utils import device, set_seeds
import configparser


config={
         "proprio_dim": 7,
        "action_dim": 2,
        "visual_embedding_dim": 256,
        "learning_rate": 3e-4,
        "weight_decay": 3e-6,
        "batch_size": 10,
    }



def train_step(policy,replay_memory):
    for s in range(1000):
        batch=replay_memory.sample(config["batch_size"])
        camera_batch,propri_batch,action_batch,feedback_batch=batch
        traning_metrics=policy.update_params(camera_batch,propri_batch,action_batch,feedback_batch)
        if s%100==0:
            print(f'step{s}, loss{traning_metrics}')
    return

policy=Policy(config).to(device)
replay_memory=torch.load('file/new_dm.dat')
train_step(policy,replay_memory)
torch.save(policy.state_dict(), 'file/policy_latest.pt')
