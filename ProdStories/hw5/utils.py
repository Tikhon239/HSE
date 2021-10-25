import ray.rllib.agents.ppo as ppo


def get_config(env_config):
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 1
    config["log_level"] = "INFO"
    config["framework"] = "torch"
    config["env"] = "Wrapper"
    config["env_config"] = env_config

    config["model"] = {
        "conv_filters": [
            [16, (3, 3), 2],
            [32, (3, 3), 2],
            [32, (3, 3), 1],
        ],
        "post_fcnet_hiddens": [32],
        "post_fcnet_activation": "relu",
        "vf_share_layers": False,
    }

    config["rollout_fragment_length"] = 100
    config["entropy_coeff"] = 0.1
    config["lambda"] = 0.95
    config["vf_loss_coeff"] = 1.0

    return config