import os
import shutil

import ray
import ray.rllib.agents.ppo as ppo
from PIL import Image
import wandb
import yaml

from wrapper import Wrapper
from utils import get_config


def train(config):
    CHECKPOINT_ROOT = config["global_config"]["CHECKPOINT_ROOT"]
    shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

    # wandb.login()
    wandb.init(config=config, **config["wandb_config"])

    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    ray.tune.register_env("Wrapper", lambda cnfg: Wrapper(**cnfg))

    agent_config = get_config(config["env_config"])
    agent = ppo.PPOTrainer(agent_config)

    for n in range(config["global_config"]["N_ITER"]):
        result = agent.train()

        wandb.log(
            {
                "episode_reward_min": result["episode_reward_min"],
                "episode_reward_mean": result["episode_reward_mean"],
                "episode_reward_max": result["episode_reward_max"],
                "episode_len_mean": result["episode_len_mean"],
            }
        )

        # sample trajectory
        if (n + 1) % 5 == 0:
            file_name = agent.save(CHECKPOINT_ROOT)
            wandb.save(file_name)

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

            gif_path = os.path.join(CHECKPOINT_ROOT, "out.gif")
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], loop=0, duration=1000 / 60)
            wandb.log({"gif": wandb.Video(gif_path, fps=30, format="gif")})


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train(config)
