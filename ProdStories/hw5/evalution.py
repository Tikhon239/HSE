import os

import ray
import ray.rllib.agents.ppo as ppo
from PIL import Image
import yaml

from wrapper import Wrapper
from utils import get_config


def evaluate(config):
    CHECKPOINT_ROOT = config["global_config"]["CHECKPOINT_ROOT"]

    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    ray.tune.register_env("Wrapper", lambda cnfg: Wrapper(**cnfg))

    agent_config = get_config(config["env_config"])
    agent = ppo.PPOTrainer(agent_config)
    agent.restore(CHECKPOINT_ROOT)

    env = Wrapper(**config["env_config"])
    obs = env.reset()

    frames = []

    for _ in range(500):
        action = agent.compute_single_action(obs)

        frame = (
            Image.fromarray(env._map.render(env._agent))
                .convert("RGB")
                .resize((500, 500), Image.NEAREST)
                .quantize()
        )
        frames.append(frame)

        obs, reward, done, info = env.step(action)
        if done:
            break

    gif_path = os.path.join(CHECKPOINT_ROOT, "evaluate.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0, duration=1000 / 60)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    evaluate(config)
