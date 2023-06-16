import torch
from common.utils import device
from env.custom_env import Custom_env
from common.models import Policy

config={
         "proprio_dim": 7,
        "action_dim": 2,
        "visual_embedding_dim": 256,
        "learning_rate": 3e-4,
        "weight_decay": 3e-6,
        "batch_size": 10,
    }
policy=Policy(config).to(device)
policy.load_state_dict(torch.load('../policy_latest.pt'))
env=Custom_env()

camera_obs,proprio_obs=env.reset()
lstm=None
for i in range(2000):
    action,lstm=policy.predict(camera_obs,proprio_obs,lstm)
    action_execute = [a*4.0 for a in action]
    next_camera_obs, next_proprio_obs, reward, done = env.step(action)
    print(action_execute)
    camera_obs, proprio_obs = next_camera_obs, next_proprio_obs



