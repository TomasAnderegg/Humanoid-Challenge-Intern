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
3. The dataset I will need to create(define one episode e.g. is it going to be a full sorting day ?) with an egocentric view using teleoperation from my leader robot with a wrist camera
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
 The success rate will be evaluated for N rollout in Mujoco. For one rollout mujoco will put the object in a random position inside the specific zone, then the instruction to the robot will be given and perform the task. I check whether the task has been completed successfully or not in a time limit. A task failed if it takes too long. I will first start with N = 20 rollouts. I will have 4 different pills. For each pill, a time limit of 20s per rollout. We will have 16 cells * N rollouts * 30s = 24000s -> ~6.7h

 So to break it down more clearly:
 1 rollout: 1 attempt by the robot = max 20s
 1 compartements with 1 pill I will have 50 rollouts = 20 * 20 s = 6.6min
 4 compartement with 4 pill: 4*4*20*20 = 1.7h

 This is only for one model. If I would like to train another model (e.g. vanilla) it is an extra 1.7h = total of ~3.6h of computing.
 




## What is Mujoco
Mujoco is a physics engine that reads a description of a world/environment from an XML file. In this file, we describe the obects such as the shape, color, mass, position etc. Mujoco will apply the physic: makes the objects fall, collide or get pushed. 

The difference objects I need to simulate:
1. My SO101 arm (exists already: https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101)
2. The table where the arm is
3. Pill organizer 
4. Pills of different colours 
5. Camera on the wrist of the arm


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



