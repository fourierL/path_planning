import time
import torch
import numpy as np
from utils import KeyboardObserver, TrajectoriesDataset
from env.custom_env import Custom_env


def main():
    env=Custom_env()
    keyboard_obs = KeyboardObserver()
    replay_memory = TrajectoriesDataset(600)
    camera_obs, proprio_obs = env.reset()
    time.sleep(2)
    print("Go!")
    episodes_count = 0
    done=False
    step=0
    while episodes_count < 20:


        start_time = time.time()
        action = np.array([0.0,0.0])
        if keyboard_obs.has_joints_cor():
            # print(f'step{step}')
            step+=1
            action = keyboard_obs.direction
            # print(action)
            next_camera_obs, next_proprio_obs,reward,done= env.step(action)
            action_save=[a/4.0 for a in action]
            print(action_save)
            replay_memory.add(camera_obs, proprio_obs, action_save, [1])
            camera_obs, proprio_obs = next_camera_obs, next_proprio_obs
        if keyboard_obs.reset_button:
            replay_memory.reset_current_traj()
            camera_obs, proprio_obs = env.reset()
            keyboard_obs.reset()
        elif done:
            replay_memory.save_current_traj()
            camera_obs, proprio_obs = env.reset()
            episodes_count += 1
            keyboard_obs.reset()
            done = False
        else:
            # loop_sleep(start_time)
            pass
    torch.save(replay_memory, '../new_dm.dat')
    return

if __name__ == "__main__":
    main()
    pass

