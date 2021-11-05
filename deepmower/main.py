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
                     state='lawnmower.lawn1.state', \
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
env = SkipFrame(env, skip=1)

### CHECK NVIDIA CUDA AVAILABILITY

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}\n")

### CHECKPOINT SAVING

save_dir = Path("../checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
#checkpoint = Path('..\\checkpoints\\2021-11-05T00-18-09\Hank_net_0.chkpt')
hank = Hank(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)#, checkpoint=checkpoint)

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

    prev_info = None
    info = None

    rewrd = 0
    propane_points = 0


    # Episode training
    while True:
        frame_count += 1
        cur_fuel_pickup = 0
        fuel_rew = 0
        rewrd = 0

        # Run agent on the state
        action = hank.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)

        # reward function information
        if prev_info is not None:
            if info["GAME_FUEL"] > prev_info["GAME_FUEL"] and frame_count > 65:
                fuel_pickups += 1
                cur_fuel_pickup = 1
                fuel_rew = cur_fuel_pickup * 100 * (1 - 1 / (1 + np.exp(-frame_count / 600)))
                print(f"Frame: {frame_count}, reward: {fuel_rew}")
            if info["DIR"] != prev_info["DIR"]:
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

        # Remember
        hank.cache(state, next_state, action, rewrd, done)

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

        # Learn
        q, loss = hank.learn()

        # Logging
        logger.log_step(rewrd, loss, q)

        # Update state
        state = next_state

        # Store previous info
        prev_info = info

        # Render frame
        env.render()

        propane_points += rewrd

        # Check if end condition is reached
        if done or info["GAME_FUEL"] == 0:
            if propane_points >= best_propane_points:
                best_propane_points = propane_points
                new_best = True
                print(f"~~~ NEW BEST!  Good job, Hank!  Top Propane Points = {best_reward}")
            break



    logger.log_episode()

    if e % 10 == 0:
        hank.save()
        logger.record(episode=e, epsilon=hank.exploration_rate, step=hank.curr_step)
