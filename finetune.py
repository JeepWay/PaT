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

from envs.register import registration_envs
from mask_pack import PPO, ACKTR
from mask_pack.common.evaluation import evaluate_policy
from mask_pack.common.callbacks import MetricsCallback
from mask_pack.common.dummy_vec_env import CustomDummyVecEnv

def train(config: Dict[str, Any], finetune_config: Dict[str, Any]):
    print(f"\n{'-' * 30}   Start Training   {'-' * 30}\n")
    vec_env: CustomDummyVecEnv = make_vec_env(
        env_id=finetune_config["env_id"], 
        n_envs=finetune_config["n_envs"], 
        monitor_dir=finetune_config['save_path'], 
        monitor_kwargs=finetune_config["monitor_kwargs"],
        env_kwargs=finetune_config["env_kwargs"],
        vec_env_cls=CustomDummyVecEnv,
    )
    if "PPO" in config['save_path']:
        model = PPO.load(os.path.join(config['test_dir'], config["env_id"]), tensorboard_log=finetune_config['save_path'], **finetune_config["PPO_kwargs"])  # need to reset ppo.clip_range
        vec_env.rebuild_obs_buf(model.observation_space, model.action_space)
        model.env = vec_env
        model.n_envs = finetune_config["n_envs"]
        model.env.seed(finetune_config["train_seed"])
        model.learn(
            total_timesteps=finetune_config["total_timesteps"], 
            progress_bar=True, 
            callback=CallbackList([
                MetricsCallback(finetune_config['save_path']),
            ])
        )
    elif "ACKTR" in config['save_path']:
        model = ACKTR.load(os.path.join(config['test_dir'], config["env_id"]), tensorboard_log=finetune_config['save_path'], **finetune_config["ACKTR_kwargs"])
        vec_env.rebuild_obs_buf(model.observation_space, model.action_space)
        model.env = vec_env
        model.n_envs = finetune_config["n_envs"]
        model.env.seed(finetune_config["train_seed"])
        model.learn(
            total_timesteps=config["total_timesteps"], 
            progress_bar=True, 
            callback=CallbackList([
                MetricsCallback(finetune_config['save_path']),
            ])
        )
    print(f"Training finished. Model saved at {finetune_config['save_path']}")
    model.save(os.path.join(finetune_config['save_path'], config["env_id"]))
    print(f"\n{'-' * 30}   Complete Training   {'-' * 30}\n")


def test(config: Dict[str, Any], finetune_config: Dict[str, Any], is_finetuned: bool = False):
    print(f"\n{'-' * 30} Start Testing On {'Finetuned' if is_finetuned else 'Pretrained'} {'-' * 30}\n")
    ep_rewards_list = []
    ep_PEs_list = []
    for i in range(config["n_eval_seeds"]):
        eval_env: CustomDummyVecEnv = make_vec_env(
            env_id=finetune_config["env_id"], 
            n_envs=1, 
            seed=int(finetune_config["eval_seed"] + i*10), 
            env_kwargs=finetune_config["env_kwargs"],
            vec_env_cls=CustomDummyVecEnv,
        )
        # must pass config["PPO_kwargs"] to reset the `self.clip_range` to the constant
        if "PPO" in config['test_dir']:
            model = PPO.load(os.path.join(finetune_config['test_dir'] if is_finetuned else config['test_dir'], 
                        config["env_id"]), **finetune_config["PPO_kwargs"])
        elif "ACKTR" in config['test_dir']:
            model = ACKTR.load(os.path.join(finetune_config['test_dir'] if is_finetuned else config['test_dir'], 
                        config["env_id"]), **finetune_config["ACKTR_kwargs"])
        eval_env.rebuild_obs_buf(model.observation_space, model.action_space)
        episode_rewards, _, episode_PEs = evaluate_policy(
            model, eval_env, 
            n_eval_episodes=finetune_config["n_eval_episodes"], 
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
    with open(os.path.join(finetune_config['test_dir'], "finetuned_eval.txt" if is_finetuned else "pretrained_eval.txt"), "w") as file:
        file.write(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")
        file.write(f"mean_PE: {mean_PE:.3f} +/- {std_PE:.3f}\n")
    print(f"\n{'-' * 30}   Complete Testing On {'Finetuned' if is_finetuned else 'Pretrained'} {'-' * 30}\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Mask BPP with PPO and ACKTR")
    parser.add_argument('--config_path', default="settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml", type=str, help="Path to the configuration file with .yaml extension.")
    parser.add_argument('--mode', default="both", type=str, choices=["train", "test", "both"], help="Mode to train or test or both of them.")
    parser.add_argument('--finetune_config', default="settings/finetune/v2_40000.yaml", type=str, help="Path to the configuration file for fine-tuning.")
    
    args = parser.parse_args()
    if not args.config_path.endswith(".yaml"):
        raise ValueError("Please specify the path to the configuration file with a .yaml extension.")

    if not args.finetune_config.endswith(".yaml"):
        raise ValueError("Please specify the path to the fine-tuning configuration file with a .yaml extension.")
    
    # read hyperparameters from the .yaml config file
    with open(args.config_path, "r") as file:
        print(f"Loading hyperparameters from: {args.config_path}")
        config = yaml.load(file, Loader=yaml.UnsafeLoader)
        
    # read hyperparameters from the fine-tuning .yaml config file
    with open(args.finetune_config, "r") as file:
        print(f"Loading hyperparameters from: {args.finetune_config}")
        finetune_config = yaml.load(file, Loader=yaml.UnsafeLoader)
        
    # set config `save_path` according to the name of the .yaml file
    config['save_path'] = os.path.join(config['log_dir'], 
        f"{config['env_id']}_{args.config_path.split('/')[-1][len('v1_'):-len('.yaml')]}"                 
    )
    finetune_config['save_path'] = os.path.join(finetune_config['log_dir'],
        f"{finetune_config['env_id']}_finetune_{config['env_id']}_{args.config_path.split('/')[-1][len('v1_'):-len('.yaml')]}"                  # remove the prefix "v1_"
    )
    os.makedirs(finetune_config['save_path'], exist_ok=True)
    
    import shutil
    shutil.copy(args.finetune_config, os.path.join(finetune_config['save_path'], args.finetune_config.split('/')[-1])) 
    
    config['test_dir'] = config['save_path']
    finetune_config['test_dir'] = finetune_config['save_path']
    
    # register custom environments
    registration_envs()

    print("\nSystem information: ")
    get_system_info(print_info=True)

    if args.mode == "both":
        train(config, finetune_config)
        test(config, finetune_config, is_finetuned=False)
        test(config, finetune_config, is_finetuned=True)
    elif args.mode == "train":
        train(config, finetune_config)
    elif args.mode == "test":
        test(config, finetune_config, is_finetuned=False)
        test(config, finetune_config, is_finetuned=True)
    else:
        raise ValueError("Invalid mode, please select either 'train' or 'test' or 'both'")
