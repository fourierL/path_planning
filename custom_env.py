import time
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.mobiles.turtlebot import TurtleBot
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np
from pyrep.objects.vision_sensor import VisionSensor

scene_file= 'file/path_planning.ttt'

class Custom_env:
    def __init__(self):
        self.pr=PyRep()
        self.pr.launch(scene_file=scene_file,headless=False)
        self.pr.start()
        self.rob=TurtleBot()
        self.cam=VisionSensor('rgb_camera')
        self.init_pose=self.rob.get_pose()
        self.target=Shape('end')
        self.target_pos=self.target.get_position()

    def reset(self):
        self.rob.set_pose(pose=self.init_pose)
        proprio_obs=self.rob.get_pose()
        cam_obs=self.cam.capture_rgb()
        cam_obs=cam_obs.transpose([2,0,1])
        self.rob.set_joint_target_velocities([0.0,0.0])
        self.pr.step()
        return cam_obs,proprio_obs

    def step(self,action):
        done=False
        reward=1.0
        self.rob.set_joint_target_velocities(action)
        for i in range(2):
            self.pr.step()
        proprio_obs = self.rob.get_pose()
        cam_obs = self.cam.capture_rgb()
        cam_obs = cam_obs.transpose([2, 0, 1])
        if np.linalg.norm(proprio_obs[:3]-self.target_pos)<0.6:
            done=True
        # print(np.linalg.norm(proprio_obs[:3]-self.target_pos))
        return cam_obs, proprio_obs,reward,done


