from typing import Dict, Any
import gymnasium as gym
import argparse
import yaml
import os
import shutil
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module='gymnasium.envs.registration')

from envs.register import registration_envs

import logging
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve items information from the environment.")
    parser.add_argument('--config_path', default="settings/main/v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml", type=str, help="Path to the configuration file with .yaml extension.")
    parser.add_argument('--output_dir', default="test_data", type=str, help="Directory to save the output data.")
    args = parser.parse_args()
    
    if not args.config_path.endswith(".yaml"):
        raise ValueError("Please specify the path to the configuration file with a .yaml extension.")
    
    # read hyperparameters from the .yaml config file
    with open(args.config_path, "r") as file:
        logging.info(f"Loading hyperparameters from: {args.config_path}")
        config = yaml.load(file, Loader=yaml.UnsafeLoader)

    args.output_dir = os.path.join(args.output_dir, config['env_id'])
    os.makedirs(args.output_dir, exist_ok=True)

    # register custom environments
    registration_envs()
    
    for i in range(config["n_eval_seeds"]):
        eval_seed = int(config["eval_seed"] + i*10)
        output_name = f"{args.output_dir}/seed{eval_seed}_{config['n_eval_episodes']}"
        
        env = gym.make(config['env_id'], **config["env_kwargs"]) 
        obs, info = env.reset(seed=eval_seed)
        logging.debug(f"episode 1, items_per_bin: {env.unwrapped.items_creator.items_per_bin}")
        logging.debug(f"episode 1, items_list: {env.unwrapped.items_creator.items_list}")
        
        all_items_list = []
        items = [list(item) for item in env.unwrapped.items_creator.items_list]
        all_items_list.append(items)

        for i in range(config['n_eval_episodes'] - 1):
            obs, info = env.reset()
            logging.debug(f"episode {i + 2}, items_per_bin: {env.unwrapped.items_creator.items_per_bin}")
            logging.debug(f"episode {i + 2}, items_list: {env.unwrapped.items_creator.items_list}")
            items = [list(item) for item in env.unwrapped.items_creator.items_list]
            all_items_list.append(items)

            logging.debug(f"all_items_list length: {len(all_items_list)}")

            with open(f"{output_name}.txt", "w") as f:
                for episode_items in all_items_list:
                    f.write(str(episode_items) + "\n")
                    
            # np.save(f"{output_name}.npy", np.array(all_items_list, dtype=object))
            
            ## load data from .npy file
            # data = np.load(f"{output_name}.npy", allow_pickle=True)
            # print(list(data))
