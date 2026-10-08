import torch
import numpy as np
from lerobot.datasets import LeRobotDataset
from utils import smolva_to_mujoco_gripper, rad_to_deg, deg_to_rad, conversion




            
if __name__ == "__main__":
    
    repo_id = "TomasAnderegg/so101_pill_sorting_merged_v2"

    # This command allow me to only download episode 0 without downloading the video
    ds = LeRobotDataset(repo_id= repo_id,episodes = [0], download_videos = False)
    
    
    #convertion to Mujoco 
    
    actions_stack = np.stack(ds.hf_dataset['action'])
    action_converted = conversion(actions_stack=actions_stack)
    # print("after conversion:", action_converted)