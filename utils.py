import os
import yaml
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def formatted_result(dir: str) -> None: 
    """
    Read the contents of eval.txt to extract the required values, 
    and then format the results into a CSV file.
    """
    csv_file = 'formatted_result.csv'
    csv_columns = ['DirName', 'Reward', 'PE']
    file_handler = open(csv_file, f"w+", newline="\n")
    writer = csv.writer(file_handler)
    writer.writerow(csv_columns)  # Write header

    def custom_key(x):
        order1 = x.split('_')[0].split('v')[0]
        if 'PPO' in x:
            order2 = 1
        elif 'ACKTR' in x:
            order2 = 2
        elif 'DQN' in x:
            order2 = 3

        order3= int(x.split('-h')[1].split('-')[0])

        if '-c' in x:
            order4 = int(x.split('-c')[1].split('-')[0])
        else:
            order4 = 0
        
        if '-n' in x:
            order5 = int(x.split('-n')[1].split('-')[0])
        else:
            order5 = 0

        if '-b' in x:
            order6 = int(x.split('-b')[1].split('-')[0])
        else:
            order6 = 0

        if '-M' in x:
            order8 = 1
            if '-Me' in x: 
                order7 = int(math.pow(10, int(x.split('-Me')[1].split('-')[0])))
            else:
                order7 = int(x.split('-M')[1].split('-')[0]) 
        elif '-R' in x:
            order8 = 2
            if '-Re' in x: 
                order7 = int(math.pow(10, int(x.split('-Re')[1].split('-')[0])))
            else:
                order7 = int(x.split('-R')[1].split('-')[0])
        else:
            order7 = 0
            order8 = 0
        
        if '-k' in x:
            order9= int(x.split('-k')[1].split('-')[0])
        else:
            order9 = 0

        if 'rA' in x:
            order10 = 1
        elif 'rC' in x:
            order10 = 2
        return (order1, order2, order3, order4, order5, order6, order7, order8, order9, order10)

    sorted_dirs = sorted(os.listdir(dir), key=custom_key)
    for dir_name in sorted_dirs:
        print(dir_name)
        eval_file_path = os.path.join(dir, dir_name, 'eval.txt')
        if os.path.isfile(eval_file_path):
            with open(eval_file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if 'reward' in line:
                        reward = line.split(': ')[-1]
                    if 'PE' in line:
                        PE = line.split(': ')[-1]
            result = [dir_name.split('2DBpp-')[-1], reward, PE]
            writer.writerow(result)


def update_yaml_files(dir):
    """
    update the YAML files according to the filename
    """
    for filename in os.listdir(dir):
        if filename.endswith(".yaml"):  # 確保處理的是YAML文件
            filepath = os.path.join(dir, filename)
            with open(filepath, 'r') as file:
                data = yaml.load(file, Loader=yaml.UnsafeLoader)  # 讀取YAML文件

            if "-F" in filename:
                data['PPO_kwargs']['add_invalid_probs'] = False
            else:
                data['PPO_kwargs']['add_invalid_probs'] = True
                
            if "v" in filename:
                if "v1" in filename:
                    data['env_id'] = '2DBpp-v1'
                    data['total_timesteps'] = 3000000
                    data['env_kwargs']['min_items_per_bin'] = 10
                    data['env_kwargs']['max_items_per_bin'] = 20
                    if "atten1" in filename: 
                        data['policy_kwargs']['network'] = "CnnAttenMlpNetwork1_v1"
                    elif "transform" in filename:
                        version = filename.split('transform')[1].split('_')[0]
                        data['policy_kwargs']['network'] = f"TransfromerNetwork{version}"
                    else:
                        data['policy_kwargs']['network'] = "CnnMlpNetwork1"
                elif "v2" in filename:
                    data['env_id'] = '2DBpp-v2'
                    data['total_timesteps'] = 6000000
                    data['env_kwargs']['min_items_per_bin'] = 15
                    data['env_kwargs']['max_items_per_bin'] = 25
                    if "atten1" in filename: 
                        data['policy_kwargs']['network'] = "CnnAttenMlpNetwork1_v1"
                    elif "transform" in filename:
                        version = filename.split('transform')[1].split('_')[0]
                        data['policy_kwargs']['network'] = f"TransfromerNetwork{version}"
                    else:
                        data['policy_kwargs']['network'] = "CnnMlpNetwork2"
                elif "v3" in filename:
                    data['env_id'] = '2DBpp-v3'
                    data['total_timesteps'] = 12500000
                    data['env_kwargs']['min_items_per_bin'] = 20
                    data['env_kwargs']['max_items_per_bin'] = 30
                    if "atten1" in filename: 
                        data['policy_kwargs']['network'] = "CnnAttenMlpNetwork1_v1"
                    elif "transform" in filename:
                        version = filename.split('transform')[1].split('_')[0]
                        data['policy_kwargs']['network'] = f"TransfromerNetwork{version}"
                    else:
                        data['policy_kwargs']['network'] = "CnnMlpNetwork3"
                elif "v4" in filename:
                    data['env_id'] = '2DBpp-v4'
                    data['total_timesteps'] = 12500000
                    data['env_kwargs']['min_items_per_bin'] = 20
                    data['env_kwargs']['max_items_per_bin'] = 30
                    if "atten1" in filename:
                        data['policy_kwargs']['network'] = "CnnAttenMlpNetwork1_v1"
                    elif "transform" in filename:
                        version = filename.split('transform')[1].split('_')[0]
                        data['policy_kwargs']['network'] = f"TransfromerNetwork{version}"
                    else:
                        data['policy_kwargs']['network'] = "CnnMlpNetwork4"
            else:
                print("Lack of environment version!\n")
                return

            if "-h" in filename:
                hidden = int(filename.split('-h')[1].split('-')[0])
                data['policy_kwargs']['network_kwargs']['hidden_dim'] = hidden
            else:
                print("Lack of hidden dimension!\n")
                return
                
            if "-c" in filename:
                clip_range = float(filename.split('-c')[1].split('-')[0]) / 10
                data['PPO_kwargs']['clip_range'] = clip_range
            else:
                print("Lack of clip range!\n")
                return
            
            if "-n" in filename:
                n_steps = int(filename.split('-n')[1].split('-')[0])
                data['PPO_kwargs']['n_steps'] = n_steps
            else:
                print("Lack of n_steps!\n")
                return
            
            if "-b" in filename:
                batch_size = int(filename.split('-b')[1].split('-')[0])
                data['PPO_kwargs']['batch_size'] = batch_size
            else:
                print("Lack of batch_size!\n")
                return

            if '-R' in filename:
                if '-Re' in filename:
                    mask_coef = int(math.pow(10, int(filename.split('-Re')[1].split('-')[0])))
                else:
                    mask_coef = int(filename.split('-R')[1].split('-')[0])
                data['policy_kwargs']['dist_kwargs']['mask_strategy'] = 'replace'
                data['policy_kwargs']['dist_kwargs']['mask_replace_coef'] = -mask_coef
            elif '-M' in filename:
                if '-Me' in filename:
                    mask_coef = int(math.pow(10, int(filename.split('-Me')[1].split('-')[0])))
                else:
                    mask_coef = int(filename.split('-M')[1].split('-')[0])
                data['policy_kwargs']['dist_kwargs']['mask_strategy'] = 'minus'
                data['policy_kwargs']['dist_kwargs']['mask_minus_coef'] = mask_coef
            else:
                print("Lack of mask strategy!\n")
                return
            
            if "-k" in filename:
                n_epochs = int(filename.split('-k')[1].split('-')[0])
                data['PPO_kwargs']['n_epochs'] = n_epochs
            else:
                print("Lack of n_epochs!\n")
                return

            if '-r' in filename:
                if '-rA' in filename:
                    data['env_kwargs']['reward_type'] = 'area'
                elif '-rC' in filename:
                    data['env_kwargs']['reward_type'] = 'compactness'
            else:
                print("Lack of reward type!\n")
                return
        
            if '-P' in filename:
                data['policy_kwargs']['mask_type'] = 'predict'
            else:
                data['policy_kwargs']['mask_type'] = 'truth'

            if "transform" in filename:
                str1 = filename.split('transform')[1].split('-')[0]
                version = str1[0]
                position_encode = True if str1[2] == 'T' else False
                use_pad_mask = True if str1[3] == 'T' else False
                d_model = int(str1.split(',')[1])
                nhead = int(str1.split(',')[2])
                dim_feedforward = int(str1.split(',')[3])
                dropout = float(str1.split(',')[4])
                num_layers = int(str1.split(',')[5])
                data['policy_kwargs']['network_kwargs']['position_encode'] = position_encode
                data['policy_kwargs']['network_kwargs']['use_pad_mask'] = use_pad_mask
                data['policy_kwargs']['network_kwargs']['transformer_kwargs']['d_model'] = d_model
                data['policy_kwargs']['network_kwargs']['transformer_kwargs']['nhead'] = nhead
                data['policy_kwargs']['network_kwargs']['transformer_kwargs']['dim_feedforward'] = dim_feedforward
                data['policy_kwargs']['network_kwargs']['transformer_kwargs']['dropout'] = dropout
                data['policy_kwargs']['network_kwargs']['transformer_kwargs']['batch_first'] = True
                data['policy_kwargs']['network_kwargs']['transformer_kwargs']['norm_first'] = False
                if version in ['1', '2']:
                    data['policy_kwargs']['network_kwargs']['num_layers'] = num_layers
                elif version in ['3', '4']:
                    data['policy_kwargs']['network_kwargs']['transformer_kwargs']['num_encoder_layers'] = num_layers
                    data['policy_kwargs']['network_kwargs']['transformer_kwargs']['num_decoder_layers'] = num_layers


            # update the YAML file
            with open(filepath, 'w') as file:
                yaml.dump(data, file, sort_keys=False)


def copy_file1(dir: str) -> None: 
    import shutil
    source_file = "main/v1_PPO-h200-c02-n64-b32-R15-atten1FT64T-atten1FT64T-k1-rA.yaml"
    destination_files = [
        "v1_PPO-h200-c02-n64-b32-M0-atten1FT64T-k1-rA.yaml",
        "v1_PPO-h200-c02-n64-b32-M7-atten1FT64T-k1-rA.yaml",
        "v1_PPO-h200-c02-n64-b32-M30-atten1FT64T-k1-rA.yaml",
        "v1_PPO-h200-c02-n64-b32-M50-atten1FT64T-k1-rA.yaml",
        "v1_PPO-h200-c02-n64-b32-M100-atten1FT64T-k1-rA.yaml",
        "v1_PPO-h200-c02-n64-b32-M500-atten1FT64T-k1-rA.yaml",
        "v1_PPO-h200-c02-n64-b32-Me3-atten1FT64T-k1-rA.yaml",
        "v1_PPO-h200-c02-n64-b32-Me4-atten1FT64T-k1-rA.yaml",
        "v1_PPO-h200-c02-n64-b32-Me5-atten1FT64T-k1-rA.yaml",
        "v1_PPO-h200-c02-n64-b32-Me6-atten1FT64T-k1-rA.yaml",
        "v1_PPO-h200-c02-n64-b32-Me7-atten1FT64T-k1-rA.yaml",
        "v1_PPO-h200-c02-n64-b32-Me8-atten1FT64T-k1-rA.yaml",
    ]
    for destination_file in destination_files:
        shutil.copy(os.path.join(dir, source_file), os.path.join(dir, destination_file))


def copy_file2(dir: str) -> None: 
    import shutil
    source_file = "main/v4_PPO-h1600-c02-n64-b32-R15-atten1FT64T-k1-rA.yaml"
    destination_files = [
        "v4_PPO-h1600-c02-n64-b32-R15-atten1FT64T-k1-rA.yaml",
        "v4_PPO-h1600-c02-n64-b64-R15-atten1FT64T-k5-rA.yaml",
        "v4_PPO-h1600-c02-n128-b32-R15-atten1FT64T-k1-rA.yaml",
        "v4_PPO-h1600-c02-n128-b64-R15-atten1FT64T-k5-rA.yaml",
        "v4_PPO-h1600-c04-n64-b32-R15-atten1FT64T-k1-rA.yaml",
        "v4_PPO-h1600-c04-n64-b64-R15-atten1FT64T-k5-rA.yaml",
        "v4_PPO-h1600-c04-n128-b32-R15-atten1FT64T-k1-rA.yaml",
        "v4_PPO-h1600-c04-n128-b64-R15-atten1FT64T-k5-rA.yaml",
        "v4_PPO-h3200-c02-n64-b32-R15-atten1FT64T-k1-rA.yaml",
        "v4_PPO-h3200-c02-n64-b64-R15-atten1FT64T-k5-rA.yaml",
        "v4_PPO-h3200-c02-n128-b32-R15-atten1FT64T-k1-rA.yaml",
        "v4_PPO-h3200-c02-n128-b64-R15-atten1FT64T-k5-rA.yaml",
        "v4_PPO-h3200-c04-n64-b32-R15-atten1FT64T-k1-rA.yaml",
        "v4_PPO-h3200-c04-n64-b64-R15-atten1FT64T-k5-rA.yaml",
        "v4_PPO-h3200-c04-n128-b32-R15-atten1FT64T-k1-rA.yaml",
        "v4_PPO-h3200-c04-n128-b64-R15-atten1FT64T-k5-rA.yaml",
    ]
    for destination_file in destination_files:
        shutil.copy(os.path.join(dir, source_file), os.path.join(dir, destination_file))


def __plot_mask_diff_strategy_coef(location, title: str, acc_replace: list, acc_minus: list):
    x_labels = ["0", "7", "15", "30", "50", "100", "500", "1.0E+03", "1.0E+04", "1.0E+05", "1.0E+06", "1.0E+07", "1.0E+08"]
    x_positions = np.arange(len(x_labels))
    y_replace = acc_replace
    y_minus = acc_minus
    plt.subplot(location)
    plt.plot(x_positions, y_replace, marker='o', markersize=8, label="Replace", color='red')
    plt.plot(x_positions, y_minus, marker='^', markersize=8, label="Minus", color='royalblue')
    plt.title(title, fontsize=16)
    plt.xlabel("coefficient", fontsize=12)
    plt.ylabel("PE", fontsize=12)
    plt.xticks(x_positions, x_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    

def plot_all_mask_diff_strategy_coef():
    plt.figure(figsize=(12, 8))
    replace_40x40 = [80, 85, 85, 84, 83, 83, 82, 82, 82, 82, 82, 82, 82]
    minus_40x40 = [0, 50, 80, 85, 85, 85, 85, 83, 83, 81, 78, 75, 73]
    __plot_mask_diff_strategy_coef(221, "(a) 10 x 10", replace_40x40, minus_40x40)
    replace_40x40 = [80, 85, 85, 84, 83, 83, 82, 82, 82, 82, 82, 82, 82]
    minus_40x40 = [0, 50, 80, 85, 85, 85, 85, 83, 83, 81, 78, 75, 73]
    __plot_mask_diff_strategy_coef(222, "(b) 20 x 20", replace_40x40, minus_40x40)
    replace_40x40 = [80, 85, 85, 84, 83, 83, 82, 82, 82, 82, 82, 82, 82]
    minus_40x40 = [0, 50, 80, 85, 85, 85, 85, 83, 83, 81, 78, 75, 73]
    __plot_mask_diff_strategy_coef(223, "(c) 40 x 40", replace_40x40, minus_40x40)
    replace_40x40 = [80, 85, 85, 84, 83, 83, 82, 82, 82, 82, 82, 82, 82]
    minus_40x40 = [0, 50, 80, 85, 85, 85, 85, 83, 83, 81, 78, 75, 73]
    __plot_mask_diff_strategy_coef(224, "(d) 32 x 50", replace_40x40, minus_40x40)
    plt.savefig(f"img/PE_diff_mask_type_coef.png")


def plot_training_curve_pe(
    dirs: dict[str, str], 
    title: str, 
    xlabel: str = 'Steps',
    ylabel: str = 'Avg Packing Efficiency',
    plot_moving_average: bool = True, 
    plot_raw: bool = True,
    window_size: int = 100,
) -> None:
    """
    Plot the training curve of packing efficiency from multiple metrics files, with option for raw or moving average.

    Args:
        dirs (dict): Dictionary mapping labels to log paths.
        title (str): Title of the plot.
        use_moving_average (bool): If True, plot moving average; if False, plot raw values.
        window_size (int): Window size for moving average calculation.
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(dirs)))  # Generate distinct colors

    for idx, (label, log_file) in enumerate(dirs.items()):
        if not os.path.exists(log_file):
            print(f"Log file {log_file} does not exist.")
            continue

        df = pd.read_csv(log_file)
        df = df.dropna(subset=['timesteps', 'ep_PE_mean'])
        df['timesteps'] = pd.to_numeric(df['timesteps'], errors='coerce')
        df['ep_PE_mean'] = pd.to_numeric(df['ep_PE_mean'], errors='coerce')
        df = df.dropna()

        timesteps = df['timesteps'].values
        ep_pe_mean = df['ep_PE_mean'].values

        if plot_moving_average:
            ep_pe_mean_ma = np.convolve(ep_pe_mean, np.ones(window_size)/window_size, mode='valid')
            timesteps_ma = timesteps[window_size-1:]
            plt.plot(timesteps_ma, ep_pe_mean_ma, label=label, color=colors[idx], linewidth=2)
        
        if plot_raw:
            plt.plot(timesteps, ep_pe_mean, color=colors[idx], linewidth=1.5, alpha=0.2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{title.replace(' ', '_')}_{ylabel.replace(' ', '_')}.png")
