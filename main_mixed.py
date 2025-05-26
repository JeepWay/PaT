from typing import Dict, Any
import gymnasium as gym
import argparse
import yaml
import os
import shutil
import numpy as np
from collections import OrderedDict
from pprint import pprint
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module='gymnasium.envs.registration')

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_system_info
from stable_baselines3.common.monitor import Monitor

from envs.register import registration_envs
from mask_pack import PPO, ACKTR
from mask_pack.common.evaluation import evaluate_policy
from mask_pack.common.callbacks import MetricsCallback
from mask_pack.common.dummy_vec_env import CustomDummyVecEnv

def train(config: Dict[str, Any]):
    print(f"\n{'-' * 30}   Start Training   {'-' * 30}\n")
    run = wandb.init(
        project=config["env_id"],
        name=config['save_path'].split("2DBpp-")[1],
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        resume = None,
    )
    def make_env_2dbpp_v1():
        monitor_path = os.path.join(config['save_path'], "v1")
        return Monitor(gym.make("2DBpp-v1", **config["env_kwargs"]), filename=monitor_path, **config["monitor_kwargs"])
    def make_env_2dbpp_v2():
        monitor_path = os.path.join(config['save_path'], "v2")
        return Monitor(gym.make("2DBpp-v2", **config["env_kwargs"]), filename=monitor_path, **config["monitor_kwargs"])
    def make_env_2dbpp_v3():
        monitor_path = os.path.join(config['save_path'], "v3")
        return Monitor(gym.make("2DBpp-v3", **config["env_kwargs"]), filename=monitor_path, **config["monitor_kwargs"])
    def make_env_2dbpp_v5():
        monitor_path = os.path.join(config['save_path'], "v5")
        return Monitor(gym.make("2DBpp-v5", **config["env_kwargs"]), filename=monitor_path, **config["monitor_kwargs"])
    def make_env_2dbpp_v6():
        monitor_path = os.path.join(config['save_path'], "v6")
        return Monitor(gym.make("2DBpp-v6", **config["env_kwargs"]), filename=monitor_path, **config["monitor_kwargs"])
    
    env_fns = [make_env_2dbpp_v3, make_env_2dbpp_v1, make_env_2dbpp_v2, make_env_2dbpp_v5, make_env_2dbpp_v6]
    vec_env = CustomDummyVecEnv(env_fns)
    
    if "PPO" in config['save_path']:
        model = PPO(env=vec_env, **config["PPO_kwargs"], tensorboard_log=config['save_path'])
        with open(os.path.join(config['save_path'], "model.txt"), "w") as f:
            f.write(f"PPO's network architecture: \n{str(model.policy)}\n")
            f.write(f"\nPPO's number of parameters: {sum(p.numel() for p in model.policy.parameters())}\n")
        model.learn(
            total_timesteps=config["total_timesteps"], 
            progress_bar=True, 
            callback=CallbackList([
                MetricsCallback(config['save_path']),
                WandbCallback(verbose=config['PPO_kwargs']['verbose']),
            ])
        )
    elif "ACKTR" in config['save_path']:
        model = ACKTR(env=vec_env, **config["ACKTR_kwargs"], tensorboard_log=config['save_path'])
        with open(os.path.join(config['save_path'], "model.txt"), "w") as f:
            f.write(f"ACKTR's network architecture: \n{str(model.policy)}\n")
            f.write(f"\nACKTR's number of parameters: {sum(p.numel() for p in model.policy.parameters())}\n")
        model.learn(
            total_timesteps=config["total_timesteps"], 
            progress_bar=True, 
            callback=CallbackList([
                MetricsCallback(config['save_path']),
                WandbCallback(verbose=config['ACKTR_kwargs']['verbose']),
            ])
        )
    print(f"Training finished. Model saved at {config['save_path']}")
    model.save(os.path.join(config['save_path'], config["env_id"]))
    # run.finish()
    print(f"\n{'-' * 30}   Complete Training   {'-' * 30}\n")


def test(config: Dict[str, Any]):
    print(f"\n{'-' * 30}   Start Testing   {'-' * 30}\n")
    ep_rewards_list = []
    ep_PEs_list = []
    for i in range(config["n_eval_seeds"]):
        eval_env = make_vec_env(
            config["env_id"], 
            n_envs=1, 
            seed=int(config["eval_seed"] + i*10), 
            env_kwargs=config["env_kwargs"],
        )
        # must pass config["PPO_kwargs"] to reset the `self.clip_range` to the constant
        if "PPO" in config['test_dir']:
            model = PPO.load(os.path.join(config['test_dir'], config["env_id"]), **config["PPO_kwargs"])
        elif "ACKTR" in config['test_dir']:
            model = ACKTR.load(os.path.join(config['test_dir'], config["env_id"]), **config["ACKTR_kwargs"])
        episode_rewards, _, episode_PEs = evaluate_policy(
            model, eval_env, 
            n_eval_episodes=config["n_eval_episodes"], 
            deterministic=True,
            return_episode_rewards=True,
        )
        ep_rewards_list.extend(episode_rewards)
        ep_PEs_list.extend(episode_PEs)

    mean_reward = np.mean(ep_rewards_list)
    std_reward = np.std(ep_rewards_list)
    mean_PE = np.mean(ep_PEs_list)
    std_PE = np.std(ep_PEs_list)
    print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"mean_PE: {mean_PE:.3f} +/- {std_PE:.3f}")
    with open(os.path.join(config['test_dir'], f"eval_{config['n_eval_episodes']}_{config['n_eval_seeds']}.txt"), "w") as file:
        file.write(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")
        file.write(f"mean_PE: {mean_PE:.3f} +/- {std_PE:.3f}\n")
    print(f"\n{'-' * 30}   Complete Testing   {'-' * 30}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Mask BPP with PPO and ACKTR")
    parser.add_argument('--config_path', default="settings/main/v2_PPO-h400-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml", type=str, help="Path to the configuration file with .yaml extension.")
    parser.add_argument('--mode', default="both", type=str, choices=["train", "test", "both"], help="Mode to train or test or both of them.")
    args = parser.parse_args()
    if not args.config_path.endswith(".yaml"):
        raise ValueError("Please specify the path to the configuration file with a .yaml extension.")

    # read hyperparameters from the .yaml config file
    with open(args.config_path, "r") as file:
        print(f"Loading hyperparameters from: {args.config_path}")
        config = yaml.load(file, Loader=yaml.UnsafeLoader)

    # set `save_path` according to the name of the .yaml file
    config['save_path'] = os.path.join(config['log_dir'], 
        f"{config['env_id']}_{args.config_path.split('/')[-1][len('v1_'):-len('.yaml')]}"                 
    )

    # set test_dir according to the mode
    if args.mode == "both":
        config['test_dir'] = config['save_path']
    elif args.mode == "test":
        config['test_dir'] = config['save_path']

    if args.mode == "both" or args.mode == "train":
        os.makedirs(config['save_path'], exist_ok=True)

        # save hyperparams
        with open(os.path.join(config['save_path'], "config.yaml"), "w") as f:
            ordered_config = OrderedDict([(key, config[key]) for key in sorted(config.keys())])
            yaml.dump(ordered_config, f)
            print("Hyperparameters for environment: ")
            pprint(ordered_config)
        
        # save command line arguments
        with open(os.path.join(config['save_path'], "args.yaml"), "w") as f:
            ordered_args = OrderedDict([(key, vars(args)[key]) for key in sorted(vars(args).keys())])
            yaml.dump(ordered_args, f)

        import shutil
        shutil.copy(args.config_path, os.path.join(config['save_path'], args.config_path.split('/')[-1])) 

    if "attention_kwargs" in config["policy_kwargs"]["network_kwargs"]:
        if config["policy_kwargs"]["network_kwargs"]["cnn_shortcut"] is True:
            assert config["policy_kwargs"]["network_kwargs"]["attention_kwargs"]["out_embed_dim"] == 64, \
                    "out_embed_dim of self-attention must be the same as the out_channels (64) of the CNN extractor, if cnn_shortcut is True."
                    
    # register custom environments
    registration_envs()

    print("\nSystem information: ")
    get_system_info(print_info=True)

    if args.mode == "both":
        train(config)
        test(config)
    elif args.mode == "train":
        train(config)
    elif args.mode == "test":
        test(config)
    else:
        raise ValueError("Invalid mode, please select either 'train' or 'test' or 'both'")
