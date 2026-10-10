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
    
    # This command allow me to only download episode 0 without downloading the video
    ds = LeRobotDataset(repo_id= repo_id,episodes = [0], download_videos = False)

    # print(ds.features)
    print(ds.fps)
    #convertion to Mujoco 
    
    actions_stack = np.stack(ds.hf_dataset['action'])
    action_converted = conversion(actions_stack=actions_stack)
    print("action converted:", len(action_converted))
    
    model = mujoco.MjModel.from_xml_path(r"/home/tjga98/mujoco_menagerie/robotstudio_so101/scene.xml") #mjModel contains model description that do not change overt time!
    # print(dir(mujoco.MjModel))
    data = mujoco.MjData(model)
    

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    duration = 3.8 #secondes
    dataset_fps = ds.fps #Hz
    hold = 2.0 #secondes

    frames = []
    times = [] 
    cmds = [] 
    actuals = [] 
    counter = 0

    mujoco.mj_resetData(model,data)
    
    data.qpos = action_converted[0] # to set initial robot pose
    data.ctrl = action_converted[0] # to set initial robot pose
    
    with mujoco.viewer.launch_passive(model, data) as viewer:

        start = time.time()
        with mujoco.Renderer(model) as renderer: #created outside the other while so it's not created every time

            while viewer.is_running() and (data.time < len(action_converted)/dataset_fps + hold):
                step_start = time.time()
                action_idx = min(int(data.time*dataset_fps), len(action_converted)-1) #because I added hold, int(data.time*dataset_fps) will give a value out of bound

                data.ctrl = action_converted[action_idx]
                # counter = counter + 1

                # print("---time diff render:", time.time() - step_start)
                mujoco.mj_step(model,data)

                if len(frames) < data.time*dataset_fps:
                    viewer.sync() #synchroniser les donnees. Quand on bouge le robot -> les valeurs sont changees aussi

                    # mujoco.mj_forward(model,data)
                    renderer.update_scene(data)
                    img = renderer.render()
                    frames.append(img)
                    # print(data.qpos)

                    # checkpoint:
                    times.append(data.time)
                    cmds.append(data.ctrl.copy())
                    actuals.append(data.qpos[:6].copy())

            # print("counter value:", counter)
            print("total time of the loop:", time.time() - start)
            print("len frames:", len(frames))


    t = np.array(times)
    cmd = np.degrees(cmds)
    act = np.degrees(actuals)
    print("len(act):", len(act))
    print("len(cmd):", len(cmd))

     #Save MP4 and Image:
    media.write_video("episode0.mp4", frames, fps=dataset_fps)
    plt.savefig("replay_ep0_tracking.png")

    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(t, cmd[:, i], label="commanded")
        plt.plot(t[:], act[:, i], label="actual")
        plt.title(f"joint {i}")
    plt.legend()
    plt.tight_layout()
    plt.show()

   
    # plt.imshow(frames); plt.axis("off"); plt.show() #To see the render in a .py file 
    # mujoco.viewer.launch(model, data)
    # mujoco.mj_kinematics(model,data) #pour reporter les shift ou modifications de l'emplacement dans le modele!
