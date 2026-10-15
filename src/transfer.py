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

RANDOM_PILLS = True     # True: random layout; False: keep the positions from the XML/keyframe
SEED = 10
FOLDED_POSITION = (0.03329552, -1.74533, 1.69, 1.30343497, -0.11124077, -0.14770727)
EPISODE_FPS = 30 #Hz
DURATION =3.8 #second
HOLD = 2 #second

SYMBOLS = ["circle", "cross", "triangle", "square"]

# def compartment_of(model, data, pill):
#     """Return the compartment the pill is in ('circle', ...), or None if it's in none."""
#     p = data.body(pill).xpos
#     for s in SYMBOLS:
#         site = model.site(f"comp_{s}")
#         centre = data.site_xpos[site.id]
#         rot = data.site_xmat[site.id].reshape(3, 3)
#         local = rot.T @ (p - centre)          # pill position in the site's own frame
#         if np.all(np.abs(local) <= model.site_size[site.id]):   # inside the box on all 3 axes
#             return s
#     return None

def transfer_to_mujoco(model, data, vla_model, pill):

    frames = []
    times = [] 
    cmds = [] 
    actuals = [] 
    input_features = {"observation.state": 
                                        {"type": "STATE", "shape": [6]},
                    "observation.images.front" : {"type": "VISUAL","shape": [3, 480,640]}
                    }
    action_converted = 0

    # 1 Rollout 
    with mujoco.viewer.launch_passive(model, data) as viewer:

        #To select the wrist camera mounted on the robot
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera("wrist_cam").id

        start = time.time()
        with mujoco.Renderer(model, height=480, width=640) as renderer: #created outside the other while so it's not created every time

            while viewer.is_running() and (data.time < 1/EPISODE_FPS + HOLD):
                step_start = time.time()
                # action_idx = min(int(data.time*EPISODE_FPS), len(action_converted)-1) #because I added hold, int(data.time*dataset_fps) will give a value out of bound
                
                mujoco.mj_step(model,data)

                if len(frames) < data.time*EPISODE_FPS:
                    
                    data.ctrl = action_converted
                    viewer.sync() #synchroniser les donnees. Quand on bouge le robot -> les valeurs sont changees aussi

                    renderer.update_scene(data, camera="wrist_cam")
                    img = renderer.render()

                    #Input of the VLA model
                    observation_state = data.qpos[:6].reshape(1,6)
                    observation_image = img
                    
                    input_features["observation.state"] = conversion(observation_state, opposite_of_rad= True) 
                    input_features["observation.images.front"] = observation_image #250x250

                    #Insert action policy VLA:
                    one_action = vla_model
                    action_converted = conversion(actions_stack=one_action)

                    frames.append(img)
                    times.append(data.time)
                    cmds.append(data.ctrl.copy())
                    actuals.append(data.qpos[:6].copy())
                # success = compartment_of(model, data, pill=pill)

            done = True
    return done
    
if __name__ == "__main__":
    
    repo_id = "TomasAnderegg/so101_pill_sorting_merged_v2"
    
    # This command allow me to only download episode 0 without downloading the video
    ds = LeRobotDataset(repo_id= repo_id,episodes = [0], download_videos = False)
    # ds = LeRobotDataset(repo_id= repo_id,episodes = [30], download_videos = False)
    # print(ds.hf_dataset["observation.state"]) #degrees
    # print(ds.meta.tasks)
    # print(ds.fps)
    #convertion to Mujoco 
    
    actions_stack = np.stack(ds.hf_dataset['action'])
    obs_stack = np.stack(ds.hf_dataset['observation.state'])
    # print(obs_stack)
    action_converted = conversion(actions_stack=actions_stack)
    obs_converted = conversion(actions_stack=obs_stack)
    # print("action converted:", len(action_converted))
    
    # # model = mujoco.MjModel.from_xml_path(r"/home/tjga98/mujoco_menagerie/robotstudio_so101/scene.xml") #mjModel contains model description that do not change overt time!
    model = mujoco.MjModel.from_xml_path(r"/home/tjga98/mujoco_menagerie/robotstudio_so101/scene_pills.xml")
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
    if RANDOM_PILLS:
       rng = np.random.default_rng(SEED)
       layout = random_pill_positions(model, data, rng)
    #    print("Pill layout:", layout)

 
    data.qpos[:6] = action_converted[0] # to set initial robot pose; [:6] corresponds to the robot joint values
    data.ctrl = action_converted[0] # to set initial robot pose
    # print("folded pose:", action_converted[0])

    print("data:", data.qpos[:6].reshape(1,6))
    # print("triangle pose:", data.site("comp_triangle").xpos)
    # print("circle pose:", data.site("comp_circle").xpos)
    # print("cross pose:", data.site("comp_cross").xpos)

    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        #To select the wrist camera mounted on the robot
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera("wrist_cam").id

        start = time.time()
        with mujoco.Renderer(model,height=480, width=640) as renderer: #created outside the other while so it's not created every time

            while viewer.is_running() and (data.time < len(action_converted)/dataset_fps + hold):
                
                step_start = time.time()
                action_idx = min(int(data.time*dataset_fps), len(action_converted)-1) #because I added hold, int(data.time*dataset_fps) will give a value out of bound
                print("circle:", data.site("comp_circle").xpos)
                data.ctrl = action_converted[action_idx]
                # print(data.ctrl[5])
                # counter = counter + 1
                
                mujoco.mj_step(model,data)

                if len(frames) < data.time*dataset_fps:
                    
                    viewer.sync() #synchroniser les donnees. Quand on bouge le robot -> les valeurs sont changees aussi
                    # print(dir(viewer))
                    # viewer.cam.type 

                    renderer.update_scene(data, camera="wrist_cam")
                    img = renderer.render()

                    # renderer.update_scene(data)
                    # img = renderer.render()
                    frames.append(img)
                    # print(data.qpos)
                    # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    # viewer.cam.fixedcamid = model.camera("wrist_cam").id
                    
                    times.append(data.time)
                    cmds.append(data.ctrl.copy())
                    actuals.append(data.qpos[:6].copy())

            # print("total time of the loop:", time.time() - start)
            # print("len frames:", len(frames))


    # t = np.array(times)
    # cmd = np.degrees(cmds)
    # act = np.degrees(actuals)
    # # print("len(act):", len(act))
    # # print("len(cmd):", len(cmd))

    #  #Save MP4 and Image:
    # media.write_video("episode0.mp4", frames, fps=dataset_fps)
    # # plt.savefig("replay_ep0_tracking.png")

    # for i in range(6):
    #     plt.subplot(3, 2, i + 1)
    #     plt.plot(t, cmd[:, i], label="commanded")
    #     plt.plot(t[:], act[:, i], label="actual")
    #     plt.title(f"joint {i}")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

   
    # plt.imshow(frames); plt.axis("off"); plt.show() #To see the render in a .py file 
    # mujoco.viewer.launch(model, data)
    # mujoco.mj_kinematics(model,data) #pour reporter les shift ou modifications de l'emplacement dans le modele!
