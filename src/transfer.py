import torch
import numpy as np
from lerobot.datasets import LeRobotDataset
from utils import smolva_to_mujoco_gripper, rad_to_deg, deg_to_rad, conversion
import time
import mujoco
import mujoco.viewer
import mediapy as media
import matplotlib.pyplot as plt
            
if __name__ == "__main__":
    
    repo_id = "TomasAnderegg/so101_pill_sorting_merged_v2"
    xml = """
        <mujoco>
            <option gravity="0 0 10" />
        <worldbody>
            <light name="top" pos="0 0 1" />
            <body name="box_and_sphere" euler="0 0 -90">
                <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
                <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
                <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
            </body>
        </worldbody>
        </mujoco>
    """
   
    

    # This command allow me to only download episode 0 without downloading the video
    ds = LeRobotDataset(repo_id= repo_id,episodes = [0], download_videos = False)
    
    
    #convertion to Mujoco 
    
    actions_stack = np.stack(ds.hf_dataset['action'])
    action_converted = conversion(actions_stack=actions_stack)
    # print("after conversion:", action_converted)
    
    model = mujoco.MjModel.from_xml_path(r"/home/tjga98/mujoco_menagerie/robotstudio_so101/scene.xml") #mjModel contains model description that do not change overt time!
    # print(dir(mujoco.MjModel))
    data = mujoco.MjData(model)

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    # print('default gravity', model.opt.gravity)
    # model.opt.gravity = (0, 0, 10)
    # print('flipped gravity', model.opt.gravity)
    
    duration = 3.8 #secondes
    framerate = 30 #Hz

    frames = []
    mujoco.mj_resetData(model,data)

    with mujoco.Renderer(model) as renderer:
        while data.time < duration:
            mujoco.mj_step(model,data)
            if len(frames) < data.time*framerate:
                # mujoco.mj_forward(model,data)
                renderer.update_scene(data)
                img = renderer.render()
                frames.append(img)  

    # plt.imshow(frames); plt.axis("off"); plt.show() #To see the render in a .py file 
    mujoco.viewer.launch(model, data)
    # mujoco.mj_kinematics(model,data) #pour reporter les shift ou modifications de l'emplacement dans le modele!
