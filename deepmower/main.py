import retro
import torch
import numpy as np
import sys
import datetime
from pathlib import Path
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from wrappers import ResizeObservation, SkipFrame, Discretizer
from metrics import MetricLogger
from hank import Hank

LAWNMOWER_LOCATION = Path().parent.absolute()
retro.data.Integrations.add_custom_path(LAWNMOWER_LOCATION)

### START ENVIRONMENT

try:
    env = retro.make(game='lawnmower',
                     state='lawn1.state',
                     inttype=retro.data.Integrations.ALL)
except FileNotFoundError:
    print(f"ERROR: lawnmower integration directory not found in the following location: {LAWNMOWER_LOCATION}")
    sys.exit()

### OBSERVATION WRAPPERS

action_space = [
    ['LEFT', 'B'],
    ['RIGHT', 'B'],
    ['DOWN', 'B'],
    ['UP', 'B']
]

env = Discretizer(env, combos=action_space)
env = ResizeObservation(env, shape=84)
env = GrayScaleObservation(env, keep_dim=False)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)
# env = SkipFrame(env, skip=1)

### CHECK NVIDIA CUDA AVAILABILITY

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}\n")

### CHECKPOINT SAVING

save_dir = Path("../checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
checkpoint = Path('..\\\checkpoints\\2021-11-08T01-18-45\\Hank_net_0.chkpt')
hank = Hank(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

### LOGGING

logger = MetricLogger(save_dir)

### BEGIN TRAINING

debug = True

episodes = 100000

best_propane_points = 0

for e in range(episodes):
    # State reset between runs
    state = env.reset()

    # Variables to keep track of for reward function
    frame_count = 0
    fuel_pickups = 0
    turns = 0

    new_best = False


    rewrd = 0
    propane_points = 0

    act = 0
    learn = 0
    delay_act = 0

    # initial action, during fuel loading
    action = hank.act(state)
    next_state, reward, done, info = env.step(action)
    prev_info = info

    game_start = 0

    frame_since_act = 0


    # Episode training
    while True:
        frame_count += 1
        frame_since_act += 1
        cur_fuel_pickup = 0
        fuel_rew = 0
        rewrd = 0

        if game_start == 0 and info["FUEL_TIME"] < 254:
            game_start = 1

        # equals 1 if action blocked, 0 if possible
        act_5fr = (prev_info["FRAME_COUNTER_5"] == 3)

        # Run agent on the state if action is possible
        if ((act == 1 and act_5fr == 0) or delay_act == 1) and game_start == 1:
            action = hank.act(state)
            frame_since_act = 0
            learn = 1  # hank should learn if he acted
            act = 0 # if acted, then acting should not occur on next frame
            delay_act = 0



        if act == 1 and act_5fr == 1:
            delay_act = 1



        # Agent performs action
        next_state, reward, done, info = env.step(action)

        # reward function information
        if prev_info is not None:
            if info["FUEL"] > prev_info["FUEL"] and frame_count > 65:
                fuel_pickups += 1
                cur_fuel_pickup = 1
                fuel_rew = cur_fuel_pickup * 100 * (1 - 1 / (1 + np.exp(-frame_count / 600)))
                print(f"Frame: {frame_count}, reward: {fuel_rew}")
            if info["DIRECTION"] != prev_info["DIRECTION"]:
                turns += 1
            else:
                turns = 0
            if info["DONE"] > prev_info["DONE"]:
                rewrd += 1

        # Penalizes for turning too much
        rewrd -= (turns - 1) * turns / 1000

        # reward for fuel pickup
        rewrd += fuel_rew

        # Penalizes for taking too long
        rewrd -= frame_count / 100000

        # Hank should only learn if he acted
        if learn == 1:
            learn = 0  # set for next episode
            hank.cache(state, next_state, action, rewrd, done)

            # Learn
            q, loss = hank.learn()

            # Logging
            logger.log_step(rewrd, loss, q)



        if debug is True:
            if e > 0:
                pass
                #print(f"Reward = {reward}")
                #print(f"Turns = {turns}")
                #print("~~~current")
                #print(info)
                #print("~~~previous")
                #print(prev_info)
                #print("~~~")
                #print("~~~")



        # Update state
        state = next_state

        # Store previous info
        prev_info = info

        # Render frame
        env.render()

        propane_points += rewrd

        # by default, no action on next possible frame


        if prev_info["PLAYER_X"] != info["PLAYER_X"] or prev_info["PLAYER_Y"] != info["PLAYER_Y"] or frame_since_act > 5:
            act = 1

        # Check if end condition is reached
        if done or info["FUEL"] == 0:
            if propane_points >= best_propane_points:
                best_propane_points = propane_points
                new_best = True
                print(f"~~~ NEW BEST!  Good job, Hank!  Top Propane Points = {best_propane_points}")
            break



        logger.log_episode()

    if e % 10 == 0:
        hank.save()
        logger.record(episode=e, epsilon=hank.exploration_rate, step=hank.curr_step)
