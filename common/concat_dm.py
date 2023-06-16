from utils import TrajectoriesDataset
import torch

def concat(dm0,dm1):
    dm_0=torch.load(dm0)
    dm_1=torch.load(dm1)
    new_dm=TrajectoriesDataset(sequence_len=100)
    new_dm.action=dm_0.action+dm_1.action
    new_dm.camera_obs=dm_0.camera_obs+dm_1.camera_obs
    new_dm.proprio_obs=dm_0.proprio_obs+dm_1.proprio_obs
    new_dm.feedback=dm_0.feedback+dm_1.feedback

    torch.save(new_dm,'new_dim.dat')
concat('demo.dat','demo1.dat')