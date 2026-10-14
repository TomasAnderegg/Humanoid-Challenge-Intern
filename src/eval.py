from transfer import transfer_to_mujoco
import torch
import numpy as np
from lerobot.datasets import LeRobotDataset
from utils import smolva_to_mujoco_gripper, rad_to_deg, deg_to_rad, conversion
import time
import mujoco
import mujoco.viewer
import mediapy as media
import matplotlib.pyplot as plt
from random_pill_pos import random_pill_positions

ROLLOUT = 10
FOLDED_POSITION = (0.03329552, -1.74533, 1.69, 1.30343497, -0.11124077, -0.14770727)
RANDOM_PILLS = True     # True: random layout; False: keep the positions from the XML/keyframe
SEED = 10

if __name__ == "__main__":

    for i in range(ROLLOUT):

        # initialize the Mujoco env
        model = mujoco.MjModel.from_xml_path(r"/home/tjga98/mujoco_menagerie/robotstudio_so101/scene_pills.xml")
        data = mujoco.MjData(model)

        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        mujoco.mj_resetData(model,data)

        # Pose init 
        # Robot initial pose :
        data.qpos[:6] = FOLDED_POSITION # to set initial robot pose; [:6] corresponds to the robot joint values
        data.ctrl = FOLDED_POSITION # to set initial robot pose
        #Pill random pose:
        if RANDOM_PILLS:
            rng = np.random.default_rng(i)
            layout = random_pill_positions(model, data, rng)
            # print("Pill layout:", layout)


        #Policy actions transfered to Mujoco
        # policy_action = 0 #when the model will be loaded
        done = False

        done = transfer_to_mujoco(model=model, data=data, vla_model=0, pill=list(layout.keys())[0])
    