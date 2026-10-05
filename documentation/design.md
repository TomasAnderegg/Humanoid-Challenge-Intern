# Robotic Medication Sorting using SmolVLA

## Context: 
For over 10 years, I've lived with my disabled aunt, she can't talk, walk properly or think as a child would. She needs extra care and my mom takes care of her. Sometimes nurses would come to my house and help my mom to do basic tasks such as wash my aunt, feed her and prepare the medication in specific boxes. But often, they make mistakes and put the wrong medication in the wrong box. This is very problematic since my aunt is epileptic. 

This gave me an idea, why not build a robotic system that could help my mom prepare the medication in the boxes.

## Break down the problem
First we need to understand how the box is, each box represents a day, each box contains 4 separations: morning, noon, evening, night. Every day, my aunt has pills she has to take but not at same period of the day: Monday she takes 2 pills in the morning and 1 at night, Tuesday only at noon etc. But the order is fixed and does not change from a week to another.

The problem is even for the nurses, arranging the pills in the right side of the box can be tricky because the pills are hard to distinguish between them. For this project, I will start easy first and then try to bring more complexity. The pills are going to be colored with different colors to be distinguishible.

Ultimately, once a box is correctly filled with the correct pills for a specific day, we need to stack all the boxes in correct order: Monday-> Sunday.

For this project, I need to establish a few things:

1. Dimensions of the different containers + Dimensions of the gripper (including error position of the wrist)
2. Which model I would use: SmolVLA because it is adapted for SO101 arm, efficient with low cost hardware and open source.
3. The dataset I will need to create(define one episode e.g. is it going to be a full sorting day ?) with an egocentric view using teleoperation from my leader robot with a front camera
4. The resource I have available
5. Post training process with the dataset
6. Implement the model in Mujoco
7. Establish the evaluation 


## Outline of the project
### Research questions
1. How can my robot help my mom to autonomously sorts some medications based on a prescription into a pill organizer for a single day ?
2. How post-training a model affects tasks performance using home made dataset.

### Hypothesis
In this project: 
1. All pills are considered in different colours in the first time to be clearly visible.
2. All pills are considered solid and compact. 


### Ablation
1. I will try to analyse the effect of the vanilla model and post-train model 


### Evaluation
 The success rate will be evaluated for N rollouts in Mujoco. One rollout corresponds to 1 episode for the trained policy, Mujoco will put the 4 coloured pills in a random position inside the specific 20x20cm area, then the instruction to the robot will be given, the policy will produce the joint value to produce the action and perform the task. At the end of a rollout, there will be a reset and the pills will be put in a random position again. I check whether a task has been completed successfully or not in a time limit for a given rollout. The test set are designed so the robot has never seen the task before.

 So to break it down more clearly:
 1 rollout = 1 episode: 1 attempt by the robot with a duration of max 45s
 There will be approaximatelly 320 rollouts: 20 rollouts per cell for all 16 cells (12 trained and 4 tests).

 I will identify 3 types of failures: 
 1. Wrong colour
 2. Wrong cell (or compartement)
 3. Dropped or missed
 


## What is Mujoco
Mujoco is a physics engine that reads a description of a world/environment from an XML file. In this file, we describe the obects such as the shape, color, mass, position etc. Mujoco will apply the physic: makes the objects fall, collide or get pushed. 

The difference objects I need to simulate:
1. My SO101 arm (exists already: https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)
2. The table where the arm is
3. Pill organizer 
4. Pills of different colours 
5. Camera on the front of the arm

## Resources and Data flow 
Since I don't have good capacities to post-train a model. I will use Google Colab for post-training. But I need to move my dataset and model between my PC and Colab. The post-training will happen in Cola but the simulation will happen on my PC. The best way I found is pushing the recorded demos in Hugging Face, load it in Colab, post-train, push again on HF and run Mujoco. My PC -> HF Hub -> Colab -> HF Hub -> My PC. 

## Calibration: sim and real mapping 
I need to get the real joint values by live reading them. The position of the arm should match the one in Mujoco. Once I have real joint values, I'll make sure they are in the same unit (radian): degree to radian to make Mujoco sim to work. I'll implement a offset transform that maps real joint values into Mujoco's with the corrected offest.

Value of min max recorded during calibration with LeRobot (Follower) (Units: tick):

| Name          |  Min |  Pos |  Max |
|---------------|-----:|-----:|-----:|
| shoulder_pan  |  688 | 1988 | 3235 |
| shoulder_lift |  828 |  844 | 3237 |
| elbow_flex    |  904 | 3107 | 3115 |
| wrist_flex    |  847 | 2875 | 3198 |
| gripper       | 2016 | 2046 | 3530 |

Same but for the leader (Units: tick): 

| Name          |  Min |  Pos |  Max |
|---------------|-----:|-----:|-----:|
| shoulder_pan  |  788 | 2096 | 3357 |
| shoulder_lift |  820 |  839 | 3214 |
| elbow_flex    |  942 | 3157 | 3166 |
| wrist_flex    |  782 | 2527 | 3109 |
| gripper       | 2032 | 2046 | 3244 |

Values in Mujoco (Units: rad):

| Name          |  Min (rad) | Pos |  Max (rad) |
|---------------|-----------:|:---:|-----------:|
| shoulder_pan  |   -1.91986 |  –  |    1.91986 |
| shoulder_lift |   -1.74533 |  –  |    1.74533 |
| elbow_flex    |   -1.69    |  –  |    1.69    |
| wrist_flex    |   -1.65806 |  –  |    1.65806 |
| gripper       |   -0.17453 |  –  |    1.74533 |

Angle range for the follower: (Units: degree)

| Name          | Real span (°) | MuJoCo span (°) | Gap (°) |
|---------------|--------------:|----------------:|--------:|
| shoulder_pan  |        223.85 |          220.00 |    3.85 |
| shoulder_lift |        211.72 |          200.00 |   11.72 |
| elbow_flex    |        194.32 |          193.66 |    0.66 |
| wrist_flex    |        206.63 |          190.00 |   16.63 |
| gripper       |        133.06 |          110.00 |   23.06 |

I need to be faithfull between the follower arm and the one in the Mujoco sim. I will take the measured values and map it to Mujoco's. The robot uses a motor with 12 bit encorder, which means a 360° corresponds to 4096 ticks. For the gripper, it is an opening of ~133° 

The tranform function from Hardware -> LeRobot Software:
f(x) = (x-val_min)/(val_max - val_min)*100 (%)
This formula gives the percentage of the joint value span.

NB: This formula only apply for the gripper the rest of the joint use degree value.

The tranform function from LeRobot Software -> Mujoco:
g_d(x) = 1.3306*x - 10 (°) 
g_r(x) = 0.02322*x - 0.174  (rad) then clip to Mujoco's interval 

Since the span of the gripper IRL is 133°. In Mujoco, it would be -10°(lower bound) + 133° = 123°. We have two points (0,-10) meaning when the gripper is closed (0) in the simluation it translated by -10 and similarly for (100, 123.06). 


I wanted to see if there are any measurement bias between the simulation and real hardware, so for a same pose of the robot, I visualize the joint values in the sim and IRL 2 times:

| Name          | Real, try 1 (°) | Real, try 2 (°) | MuJoCo (°) | \|try 2 − try 1\| (°) |
|---------------|----------------:|----------------:|-----------:|----------------------:|
| shoulder_pan  |           -2.95 |            0.75 |          0 |                   3.7 |
| shoulder_lift |           -1.93 |           -0.35 |          0 |                   1.6 |
| elbow_flex    |            6.68 |            3.34 |          0 |                   3.3 |
| wrist_flex    |            2.77 |            4.00 |          0 |                   1.2 |
| wrist_roll    |           -0.22 |            3.56 |          0 |                   3.8 |

Based on those observations, I concluded that there is no significant bias between the real world and Mujoco except for the gripper. During the hardware calibration (checking the min and max range for the differents joint value) of the robot, the closed gripper represents 0, whereas in Mujoco the value 0 for the gripper does not correspond to a fully closed but slgithly open which create. By replicating the same pose than in Mujoco 2 times I got: 9.57% and  9.41% IRL corresponding to 0° in Mujoco. By applying the formula g_d(x=9.41)= 2.5° which can be neglected because the larger difference according to the table is ~4° and the gripper bias is lower. 
Sign check: startinf from pose A, I moved each arm joint on the real robot, then matched it in the Mujoco viewer. The sim value had the same sign for all 5 joints.

I'll later compare the recorded camera video with the Mujoco replay video side by side. If the poses differ at the same moement, it means my mapping (unit, sign or gripper formula) is wrong for that joint.

Suggestion: After converting the whole episode, you count how many values fall outside each joint's ctrlrange. (MuJoCo would silently clip
    them.)

  Written as a sentence:

  ▎ "I'll compare every converted value with its joint's ctrlrange. If many are outside, it means the mapping is wrong, or the real
  ▎ arm went beyond the sim's limits (possible for wrist_flex and the gripper, given my span table)."


## Dataset protocol

1. Setup: The robot will be on a black table, in front of it there will be a zone of area 20x20 cm^2 marked with white tape. The same environment will be reproduce in Mujoco
2. The only camera will be the egocentric camera of the robot located at the front.
3. The start position the robot will be the rest pose in Mujoco where all joint values are in 0. The pill are place in a random order inside the area. The pills cannot be one on top of the other but only next to at most.
4. An episode is done when the task is completed e.g. the pill is in the right container.
5. The instruction protocol will follow the same structure: "Put the {colour} pill in the {compartment} compartment."
6. The time: 30s of motion, plus 15s to put the pill = at least 45s. Having enough variety for the policy to learn. SmolVLA docs talk about ~50 episodes for a single task when fine-tuning. Since it is the same task for me which is pick and place.
7. Run a small pilot test 
## Questions for myself
Q1: The scale problem. Do the numbers.

R1: I will need to measure the pill, compartment and the SO101 gripper maximum opening in mm.
The pill compartment measures: 10.8 cm de lenght, 4.5 cm de width, 1.5 cm wall height. Each compartement is separated with a distance of 2.5 cm. The pill size: 2 cm de lenght, 0.5 cm de width et 0.5 cm wall height. The max opening of the gripper: 15 cm.

Q2: The grasp. Watch yourself pick up a pill. How many fingers do you use, and how? Then compare that with a two-finger
  parallel gripper.
  → What does your retargeting turn a fingertip pinch into? Is "gripper closed" when thumb and index finger touch a good
  enough rule for a 7 mm object?

R2: Since I will use a leader arm to teleoperate and grab the pill, I won't need to worry about fingers.

Q3: The simulator. Does Mujoco or LIBERO contain a pill organizer, or anything close to one?
  → If not, what would you have to build, and how hard is it in MuJoCo with simple shapes (cylinders, a grid of boxes)? Does
  that change your answer to Q2 from last time (LIBERO vs. a simpler sim)?

R3: There is no pill organizer to the best of my knowledge. I will need to construct it. The easiest thing to do is using Mujoco and build it there. I have always wanted to try Mujoco Simulator. Hopefully it won't need to much time, I am aiming at spending no more than a day and half.

Q4: What is one episode? "Sort a whole week" means 7 days × 2–4 times × several pills: a very long horizon.
  → What's the smallest unit you'd record and evaluate? And how does the robot know what to do: one instruction per pill, or
  a whole "prescription"?

R4: Not sure how hard this project is going to be for only a week, I will keep it simple so 1 episode represents 1 pill -> right container at the right side. And also for simplicity, I will start by only a day of sorting. If I have time, I will add another day. 

Q5: Generalization with a small dataset. Say you have 3 pill types and 7 compartments.
  → How many (pill, compartment) combinations is that? Do you need to record all of them? What would it show if the robot
  succeeded on a combination it never saw during training? (This is likely your answer to Q3 from last time: the thing that
  makes a reviewer read more carefully.)

R5: According to the task description, the composition is fixed by the description during training. The idea would be to create a small dataset with different combination according to the task description: I will start by one pill, different combination then progressively scale up the number of pills up to 4 with different combinations. The idea is to use at inference the robot trained on a zero shot full new description, recognize the pull and able to place it in the right side of the box.

Imagine 3 pills colours (Red, blue, yellow) and 4 compartments (Morning, Noon, Evening and Night). I will record demos only for the train cells for example for red: Morning(train), Noon(train), Evening(train) and Night(test). During test time, I will give an instruction the robot has never seen before "put the red pill in the night compartment". This will allow me to evaluate if the robot hasn't just memorized the motion but understood red and night.


## Notes
- I will need to define a fixed area during the teleoperation part where I will record the object. The robot will (hopefully) succeed because the object is inside it. If the object is outside the fixed area, it will mostly fail. The area will be 20 cm lenght and 20 cm width.
- The XML of the SO101 arm, the actuator value of the range are in radian, but the files in lerobot for the so_follower are in degree ! Carefull (/home/tjga98/Humanoid-Challenge-Intern/lerobot/src/lerobot/robots/so_follower/config_so_follower.py). During calibration, I have to be carefull to match mujoco's and also the values on my hardware.
- By default, MJCF files reads angles in degrees, but in this file it is in radian because of <compiler angle="radian" meshdir="assets" autolimits="true"/>
- One round will be composed of 4 episodes where each episode corresponds to a specific pill. At the begining, the 4 pills are randomly placed on the area, the robot will perform the 4 tasks. At the end of a round none pills should be on the table but inside their respective containers. Then a reset will be done by shuffling the pills and randomly placed on the area.
- During training, the task will be randomly selected so the robot does not learn a rule concerning the order of the pill.
- There will be a test cell, where during training the combination with that cell for a specific colour pill won't be avaible to test the model understanding.

-A compartement can have 4 pills at the same time this allows to have more data diversity (3^4=81 possible rounds).  

- For each colour pill, I will record 8 times a task e.g. Put the red pill in the morning compartemen, for this task I will record 8 times the pill in random position in the area, and execute the  task. I will have for a specific pill and all the train cell: 3x8 = 24 recordings. Since I have 4 pills, this makes a total of 4x24recordings = 96 episdodes recorded for the train. Each recording lasts 45s, meaning the dataset is 1h15 long.

- Because of the quality of the camera, the texts (morning, noon, evening, night) are not visible well enough to recognize, so I change them by symbols: (morning -> circle, noon-> cross, evening -> triangle, night -> square). It is much easier to recognize on camera. 

- To compose a round = 4 episodes from balanced pill task. Each round uses one train cell per colour, chosen so that after 24 rounds, every train cell has appeared exactly 8 times. The order of the 4 episodes inside a round is random. A total of 24 rounds (=96/4) will be created. Knowing that a round = 4x45s=3min, 24 rounds will last 1h12 for the train.

## Annexe
Prompt instruction for training:

Put the red pill in the circle compartment.
Put the red pill in the cross compartment.
Put the red pill in the triangle compartment.
Put the blue pill in the circle compartment.
Put the blue pill in the triangle compartment.
Put the blue pill in the square compartment.
Put the yellow pill in the cross compartment.
Put the yellow pill in the triangle compartment.
Put the yellow pill in the square compartment.
Put the green pill in the circle compartment.
Put the green pill in the cross compartment.
Put the green pill in the square compartment.

Prompt instruction for test:

Put the red pill in the square compartment.
Put the blue pill in the cross compartment.
Put the yellow pill in the circle compartment.
Put the green pill in the triangle compartment.

## References
1. https://mujoco.readthedocs.io/en/stable/python.html
2. https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotstudio_so101/so101.xml
3. https://huggingface.co/docs/lerobot/en/installation
4. https://arxiv.org/abs/2506.01844

