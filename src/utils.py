import math
import numpy as np

#correct order in my data + Mujoco's order
SHOULDER_PAN_IDX = 0
SHOULDER_LIFT_IDX = 1
ELBOW_FLEX_IDX = 2
WRIST_FLEX_IDX = 3
WRIST_ROLL_IDX = 4
GRIPPER_IDX = 5

#Value actuators min max from mujoco so101.xml
SHOULDER_PAN_MIN_VAL =-1.91986 
SHOULDER_PAN_MAX_VAL =1.91986
SHOULDER_LIFT_MIN_VAL = -1.74533 
SHOULDER_LIFT_MAX_VAL = 1.74533
ELBOW_FLEX_MIN_VAL = -1.69 
ELBOW_FLEX_MAX_VAL = 1.69
WRIST_FLEX_MIN_VAL = -1.65806 
WRIST_FLEX_MAX_VAL = 1.65806
WRIST_ROLL_MIN_VAL = -2.74385 
WRIST_ROLL_MAX_VAL = 2.84121
# joints = [SHOULDER_PAN_MIN_VAL, SHOULDER_PAN_MAX_VAL, SHOULDER_LIFT_MIN_VAL, SHOULDER_LIFT_MAX_VAL, ELBOW_FLEX_MIN_VAL, ELBOW_FLEX_MAX_VAL, WRIST_FLEX_MIN_VAL, WRIST_FLEX_MAX_VAL]
GRIPPER_MIN_VAL =-0.17453 
GRIPPER_MAX_VAL =1.74533

def deg_to_rad(deg):
    return deg * math.pi / 180.0

def rad_to_deg(rad):
    return rad * 180.0 /math.pi 

def smolva_to_mujoco_gripper(x):
    
    val_x_deg = 1.3306 * x - 10
    val_x_rad = deg_to_rad(val_x_deg)
    val_x_rad = np.clip(val_x_rad, -0.17453, 1.74533)
    return val_x_rad

def sanity_check(idx, x):
    if idx == SHOULDER_PAN_IDX and (x < SHOULDER_PAN_MIN_VAL or x > SHOULDER_PAN_MAX_VAL):
        print("Attention, out of Mujoco's bound for shoulder pan")

    if idx == SHOULDER_LIFT_IDX and (x < SHOULDER_LIFT_MIN_VAL or x > SHOULDER_LIFT_MAX_VAL):
        print("Attention, out of Mujoco's bound for shoulder lift")

    if idx == ELBOW_FLEX_IDX and (x < ELBOW_FLEX_MIN_VAL or x > ELBOW_FLEX_MAX_VAL):
        print("Attention, out of Mujoco's bound for elbow flex")

    if idx == WRIST_FLEX_IDX and (x < WRIST_FLEX_MIN_VAL or x > WRIST_FLEX_MAX_VAL):
        print("Attention, out of Mujoco's bound for wrist flex")
    
    if idx == WRIST_ROLL_IDX and (x < WRIST_ROLL_MIN_VAL or x > WRIST_ROLL_MAX_VAL):
        print("Attention, out of Mujoco's bound for wrist roll")

    if idx == GRIPPER_IDX and (x < GRIPPER_MIN_VAL or x > GRIPPER_MAX_VAL):
        print("Attention, out of Mujoco's bound for gripper")

def conversion(actions_stack):
    #Convert values into Mujoco's value: joints [0,4]: deg to rad and gripper[5] percentage to rad
    
    print("action value:", actions_stack[0])
    print (actions_stack.shape)
    row, column = actions_stack.shape

    action_to_mujoco = np.zeros((row, column))
    for i in range(row):
        for j in range(column):
            if j == GRIPPER_IDX:
                action_to_mujoco[i][j]= smolva_to_mujoco_gripper(actions_stack[i][j])
                # sanity_check(j, action_to_mujoco[i][j])
            else: 
                # sanity_check(j, action_to_mujoco[i][j])
                action_to_mujoco[i][j] = deg_to_rad(actions_stack[i][j])
                
    
    return action_to_mujoco 