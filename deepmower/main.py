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

""" CHECK NVIDIA CUDA AVAILABILITY """

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}\n")

""" START ENVIRONMENT """

try:
    save_states = [f'lawn{x}.state' for x in range(10, 0, -1)]
    env = retro.make(game='lawnmower',
                     state=save_states.pop(), # pops off lawn1.state
                     inttype=retro.data.Integrations.ALL)
except FileNotFoundError:
    print(f"ERROR: lawnmower integration directory not found in the following location: {LAWNMOWER_LOCATION}")
    sys.exit()

""" OBSERVATION WRAPPERS """

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

""" CHECKPOINT SAVING """

save_dir = Path("../checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
checkpoint = Path('..\\\checkpoints\\2021-11-10T22-13-01\\Hank_net_13.chkpt')
hank = Hank(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)

### Set these if you want it to begin learning anew with the current nn
#hank.exploration_rate = 1
#hank.exploration_rate_decay = 0.99999975

""" LOGGING """

logger = MetricLogger(save_dir)

""" BEGIN TRAINING """

debug = True
episodes = 100000
best_propane_points = 0  # aka best_cumulative_reward

for e in range(episodes):

    # State reset between runs
    init_state = env.reset()

    # Variables to keep track of for reward function
    frame_count = 0
    frame_since_act = 0
    frame_since_OOF = 0
    fuel_pickups = 0
    turns = 0
    propane_points = 0  # aka cumulative_reward

    reward = 0

    act = False
    learn = False
    delay_act = False
    game_start = False
    # new_best = False # not used

    # initial action
    action = hank.act(init_state)
    action_state = init_state  # current state when action is performed
    next_state, _, done, info = env.step(action)
    prev_info = info

    # Episode training
    while True:

        """ FRAME SENSITIVE CONDITIONS """

        frame_count += 1
        frame_since_act += 1
        # cur_fuel_pickup = 0
        fuel_rew = 0


        if not game_start and info["FUEL_TIME"] < 254: # FUEL_TIME changes randomly
            game_start = True

        # equals True if action blocked, False if possible
        act_5fr = prev_info["FRAME_COUNTER_5"] == 3

        # Run agent on the state if action is possible
        if ((act and not act_5fr) or delay_act) and game_start:
            # Hank is about to act.  Learn from prior actions

            hank.cache(action_state, next_state, action, reward, done)

            # Learn
            q, loss = hank.learn()
            propane_points += reward

            # Logging
            logger.log_step(reward, loss, q)

            reward = 0

            # Perform new action
            action = hank.act(next_state)
            action_state = next_state  # current state when action is performed
            frame_since_act = 0

            act = False  # if acted, then acting should not occur on next frame
            delay_act = False

        if act and act_5fr:
            delay_act = True

        # Agent performs action
        next_state, _, done, info = env.step(action)

        # by default, no action on next possible frame
        if (prev_info["PLAYER_X"] != info["PLAYER_X"] or 
            prev_info["PLAYER_Y"] != info["PLAYER_Y"] or 
            frame_since_act > 5
        ):
            act = True

        """ REWARD FUNCTION INFORMATION """

        ### TODO: clean up reward section

        if prev_info is not None:
            if info["FUEL"] > prev_info["FUEL"]:
                fuel_pickups += 1
                # cur_fuel_pickup = 1
                fuel_rew = 1 * 100 * (1 - 1 / (1 + np.exp(-frame_count / 600)))
                frame_since_OOF = 0
                #print(f"Frame: {frame_count}, reward: {fuel_rew}")
            if info["DIRECTION"] != prev_info["DIRECTION"]:
                turns += 1
            else:
                turns = 0
            if info["GRASS_LEFT"] < prev_info["GRASS_LEFT"]:
                reward += 1

            # Penalize for OOF'ing
            if frame_since_OOF > 3:
                reward -= 25

        # Penalizes for turning too much
        reward -= (turns - 1) * turns / 100

        # reward for fuel pickup
        reward += fuel_rew  # I think this is way too massive right now

        # Penalizes for taking too long
        reward -= (frame_since_act + 1) / 100

        """ STATE UPDATES """



        # Update state
        # state = next_state  # irrelevant now?

        # Store previous info
        prev_info = info

        # Render frame
        env.render()

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

        """ DONE CONDITIONS """

        # Hacky way to handle OOF'ing
        if info["FUEL"] == 0:
            frame_since_OOF += 1

        # Check if OOF
        if frame_since_OOF > 3 or info["GRASS_LEFT"] < 1:
            if info["GRASS_LEFT"] < 1:
                reward += 25  # maybe remove this?

            # Learn from final actions
            hank.cache(action_state, next_state, action, reward, done)

            # Learn
            q, loss = hank.learn()
            propane_points += reward

            # Logging
            logger.log_step(reward, loss, q)

            if propane_points < best_propane_points:
                print(f"Run {e} - Propane Points = {round(propane_points,1)}  ||  Top Propane Points = {round(best_propane_points,1)}")
            elif propane_points >= best_propane_points:
                best_propane_points = propane_points
                # new_best = True # not used
                print(f"Run {e} ~~~ NEW BEST!  Good job, Hank!  New Top Propane Points = {round(best_propane_points,1)}")
            break

    logger.log_episode()

    """ SAVING & CHANGING LAWNS"""

    if e % 10 == 0:
        hank.save()
        logger.record(episode=e, epsilon=hank.exploration_rate, step=hank.curr_step)

    if info["GRASS_LEFT"] < 1 and save_states:
        hank.save()
        logger.record(episode=e, epsilon=hank.exploration_rate, step=hank.curr_step)
        env.load_state(save_states.pop(), inttype = retro.data.Integrations.ALL)
    elif not save_states:
        sys.exit("HANK, YOU DID IT! YOU RAN THE GAUNTLET! LAWN 1-10 COMPLETE.")