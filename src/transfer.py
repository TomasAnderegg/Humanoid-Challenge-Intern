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
    
    
    times, cmds, actuals = [], [], []
    # This command allow me to only download episode 0 without downloading the video
    ds = LeRobotDataset(repo_id= repo_id,episodes = [0], download_videos = False)

    # print(ds.features)
    print(ds.fps)
    #convertion to Mujoco 
    
    actions_stack = np.stack(ds.hf_dataset['action'])
    action_converted = conversion(actions_stack=actions_stack)
    # print("action converted:", action_converted[0])
    
    model = mujoco.MjModel.from_xml_path(r"/home/tjga98/mujoco_menagerie/robotstudio_so101/scene.xml") #mjModel contains model description that do not change overt time!
    # print(dir(mujoco.MjModel))
    data = mujoco.MjData(model)
    

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    duration = 3.8 #secondes
    dataset_fps = ds.fps #Hz
    hold = 2.0 #secondes

    frames = []
    mujoco.mj_resetData(model,data)
    
    data.qpos = action_converted[0] # to set initial robot pose
    data.ctrl = action_converted[0] # to set initial robot pose
    
    with mujoco.viewer.launch_passive(model, data) as viewer:

        start = time.time()
        # print("start:", start)
        while viewer.is_running() and time.time() - start < 30:
            step_start = time.time()
            action_idx = int(data.time*dataset_fps)
            times.append(data.time)
            cmds.append(data.ctrl.copy())
            actuals.append(data.qpos[:6].copy())

            with mujoco.Renderer(model) as renderer:
                while data.time < len(action_converted)/dataset_fps + hold:
                    # print(data.time)
                    mujoco.mj_step(model,data)
                    viewer.sync() #synchroniser les donnees. Quand on bouge le robot -> les valeurs sont changees aussi

                    if len(frames) < data.time*dataset_fps:
                        # mujoco.mj_forward(model,data)
                        renderer.update_scene(data)
                        img = renderer.render()
                        print(data.qpos)
                        times.append(data.time)
                        cmds.append(data.ctrl.copy())
                        frames.append(img)
                    time_until_next_step = model.opt.timestep - (time.time() - step_start)
                    # print("opt.timestep:", model.opt.timestep, ",time:", time.time(), ",step start:", step_start, "---le moins:", time.time() - step_start, "le tout:", time_until_next_step)

                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)    
                # print("after start:", time.time(), "----difference:", time.time() - start)
    t = np.array(times)
    cmd = np.degrees(cmds)
    act = np.degrees(actuals)

    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(t, cmd[:, i], label="commanded")
        plt.plot(t, act[:, i], label="actual")
        plt.title(f"joint {i}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.imshow(frames); plt.axis("off"); plt.show() #To see the render in a .py file 
    # mujoco.viewer.launch(model, data)
    # mujoco.mj_kinematics(model,data) #pour reporter les shift ou modifications de l'emplacement dans le modele!
