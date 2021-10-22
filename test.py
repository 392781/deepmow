import os
import retro
from nes_py.wrappers import JoypadSpace
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
retro.data.Integrations.add_custom_path(
    SCRIPT_DIR
)
print("lawnmower" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
env = retro.make(game="lawnmower", 
    state='lawnmower.lawn1.state', \
    inttype=retro.data.Integrations.ALL)

env = JoypadSpace(
    env,
    [['right'],
    ['left'],
    ['up'],
    ['down']]
)

env.reset()
next_state, reward, done, info = env.step(action=0)
print(f'{next_state.shape},\n {reward},\n {done},\n {info}')