import gameworld.envs  # this import triggers the register() calls
import gymnasium

num_steps = 500

games = [
    "Aviate",
    "Bounce",
    "Cross",
    "Drive",
    "Explode",
    "Fruits",
    "Gold",
    "Hunt",
    "Impact",
    "Jump",
]


renders = {}
for game in games:
    env = gymnasium.make(f"Gameworld-{game}-v0")

    obs, info = env.reset()
    all_obs = [obs]

    for t in range(num_steps):
        # random actions as example
        action = env.action_space.sample()

        # step env
        obs, reward, done, truncated, info = env.step(action)

        # reset when done
        if done:
            obs, info = env.reset()

        all_obs.append(obs)

    renders[game] = all_obs

import mediapy as mp
from pathlib import Path

Path("./gameworld_videos").mkdir(parents=True, exist_ok=True)

with mp.set_show_save_dir("./gameworld_videos"):
    # Show videos
    mp.show_videos(renders, codec="gif", width=200)