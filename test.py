import os
import retro
import gym
from gym_video_streamer import VideoStreamer
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace
from wrappers import ResizeObservation, SkipFrame
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
retro.data.Integrations.add_custom_path(
    SCRIPT_DIR
)
print("lawnmower" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
env = retro.make(game="lawnmower", 
    state='lawnmower.lawn1.state', \
    inttype=retro.data.Integrations.ALL, \
    use_restricted_actions=retro.Actions.DISCRETE)

env = JoypadSpace(
    env,
    [['right', 'A'],
    ['up', 'A'],
    ['left', 'A'],
    ['down', 'A']]
)

env = GrayScaleObservation(env, keep_dim=False)
#env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
env = FrameStack(env, num_stack=4)
# env = VideoStreamer(env)

env.reset()
for e in range(40000):

    state = env.reset()
    while True:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action=action)
        env.render()
        state = next_state
        print(f'{next_state.shape},\n {reward},\n {done},\n {info}')
        if done or info['GAME_FUEL'] == 0:
            break