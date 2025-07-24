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

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_system_info

import sys
import pat
# the previous version of the package is named `mask_pack`, but now it is `pat`
sys.modules['mask_pack'] = pat 

from envs.register import registration_envs
from pat import PPO
from pat.common.evaluation import evaluate_policy
from pat.common.callbacks import MetricsCallback
from pat.common.dummy_vec_env import CustomDummyVecEnv


def test(orig_config: Dict[str, Any], target_config: Dict[str, Any]):
    print(f"\n{'-' * 30} Start Testing {'-' * 30}\n")
    ep_rewards_list = []
    ep_PEs_list = []
    for i in range(target_config["n_eval_seeds"]):
        eval_env: CustomDummyVecEnv = make_vec_env(
            env_id=target_config["env_id"], 
            n_envs=1, 
            seed=int(target_config["eval_seed"] + i*10), 
            env_kwargs=target_config["env_kwargs"],
            vec_env_cls=CustomDummyVecEnv,
        )
        # must pass config["PPO_kwargs"] to reset the `self.clip_range` to the constant
        if "PPO" in orig_config['test_dir']:
            model = PPO.load(os.path.join(orig_config['test_dir'], 
                        orig_config["env_id"]), **target_config["PPO_kwargs"])
        else: 
            raise ValueError(f"Unsupported algorithm in {orig_config['test_dir']}")
       
        eval_env.rebuild_obs_buf(model.observation_space, model.action_space)
        episode_rewards, _, episode_PEs = evaluate_policy(
            model, eval_env, 
            n_eval_episodes=target_config["n_eval_episodes"], 
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
    with open(os.path.join(target_config['test_dir'], f"eval_{target_config['n_eval_episodes']}_{target_config['n_eval_seeds']}.txt"), "w") as file:
        file.write(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")
        file.write(f"mean_PE: {mean_PE:.3f} +/- {std_PE:.3f}\n")
    print(f"\n{'-' * 30}   Complete Testing {'-' * 30}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generalization Mechanism")
    parser.add_argument('--orig_config_path', default="settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml", type=str, help="Path to the configuration file for original (large) env")
    parser.add_argument('--mode', default="test", type=str, choices=["test"], help="Only test mode")
    parser.add_argument('--target_config_path', default="settings/general/v1_10000.yaml", type=str, help="Path to the configuration file for target (small) env")
    
    args = parser.parse_args()
    if not args.orig_config_path.endswith(".yaml"):
        raise ValueError("Please specify the path to the configuration file with a .yaml extension for the original (large) environment.")

    if not args.target_config_path.endswith(".yaml"):
        raise ValueError("Please specify the path to the configuration file with a .yaml extension for the target (small) environment.")

    # read hyperparameters from the original config file
    with open(args.orig_config_path, "r") as file:
        print(f"Loading hyperparameters from: {args.orig_config_path}")
        orig_config = yaml.load(file, Loader=yaml.UnsafeLoader)

    # read hyperparameters from the target config file
    with open(args.target_config_path, "r") as file:
        print(f"Loading hyperparameters from: {args.target_config_path}")
        target_config = yaml.load(file, Loader=yaml.UnsafeLoader)
        
    # set config `save_path` according to the name of the .yaml file
    orig_config['save_path'] = os.path.join(orig_config['log_dir'],
        f"{orig_config['env_id']}_{args.orig_config_path.split('/')[-1][len('v1_'):-len('.yaml')]}"
    )
    target_config['save_path'] = os.path.join(target_config['log_dir'],
        f"{target_config['env_id']}_general_{orig_config['env_id']}_{args.orig_config_path.split('/')[-1][len('v1_'):-len('.yaml')]}"
    )
    os.makedirs(target_config['save_path'], exist_ok=True)

    import shutil
    shutil.copy(args.target_config_path, os.path.join(target_config['save_path'], args.target_config_path.split('/')[-1]))

    orig_config['test_dir'] = orig_config['save_path']
    target_config['test_dir'] = target_config['save_path']

    # register custom environments
    registration_envs()

    print("\nSystem information: ")
    get_system_info(print_info=True)

    if args.mode == "test":
        test(orig_config, target_config)
    else:
        raise ValueError("Invalid mode, please select either test")
