import os
import yaml
import csv
import math
import matplotlib.axes as axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def formatted_result(
    root_dir: str, 
    test_filename: str = 'eval_50_20.txt',
    save_filename: str = 'testing_result.csv'
) -> None: 
    """
    讀取目錄底下的所有子目錄，尋找指定的測試文件 test_filename，
    並從中提取獎勵和 PE 的平均值和標準差。
    """
    csv_columns = ['Dir_Name', 'Reward_Mean', 'Reward_Std', 
                   'PE_Mean', 'PE_Std', 
                   'PE_Percent_Mean', 'PE_Percent_Std']
    
    with open(save_filename, "w+", newline="\n") as file_handler:
        writer = csv.writer(file_handler)
        writer.writerow(csv_columns)

        for dirpath, sub_dirnames, filenames in os.walk(root_dir):
            logging.debug(f"Directory: {dirpath}")
            logging.debug(f"Subdirectories: {sub_dirnames}")
            logging.debug(f"Files: {filenames}\n")
            
            if test_filename in filenames:
                test_file_path = os.path.join(dirpath, test_filename)
                reward_mean = reward_std = pe_mean = pe_std = pe_percent_mean = pe_percent_std = "N/A"
                
                with open(test_file_path, 'r') as file:
                    for line in file:
                        if 'reward' in line:
                            try:
                                val, std = map(float, line.split(':')[-1].strip().split('+/-'))
                                reward_mean = round(val, 6)
                                reward_std = round(std, 6)
                            except:
                                pass
                        elif 'PE' in line:
                            try:
                                val, std = map(float, line.split(':')[-1].strip().split('+/-'))
                                pe_mean = round(val, 6)
                                pe_std = round(std, 6)
                                pe_percent_mean = round(val * 100, 2)
                                pe_percent_std = round(std * 100, 2)
                            except:
                                pass

                dir_name = os.path.basename(dirpath)
                result = [
                    dir_name,
                    reward_mean, reward_std,
                    pe_mean, pe_std,
                    pe_percent_mean, pe_percent_std
                ]
                writer.writerow(result)
    logging.info(f"Complete writing test results into {save_filename}.")


def update_yaml_files(root_dir: str) -> None:
    """
    根據 YAML 文件的名稱更新 YAML 文件的內容。
    """
    for filename in os.listdir(root_dir):
        if filename.endswith(".yaml"):
            filepath = os.path.join(root_dir, filename)
            with open(filepath, 'r') as file:
                data = yaml.load(file, Loader=yaml.UnsafeLoader)  # 要用 UnsafeLoader
                
            if "v" in filename:
                if "v1" in filename:
                    data['env_id'] = '2DBpp-v1'
                    data['total_timesteps'] = 3000000
                    data['env_kwargs']['min_items_per_bin'] = 10
                    data['env_kwargs']['max_items_per_bin'] = 20
                    if "transform" in filename:
                        version = filename.split('transform')[1].split('_')[0]
                        data['policy_kwargs']['network'] = f"TransfromerNetwork{version}"
                    else:
                        data['policy_kwargs']['network'] = "CnnMlpNetwork1"
                elif "v2" in filename:
                    data['env_id'] = '2DBpp-v2'
                    data['total_timesteps'] = 6000000
                    data['env_kwargs']['min_items_per_bin'] = 15
                    data['env_kwargs']['max_items_per_bin'] = 25
                    if "transform" in filename:
                        version = filename.split('transform')[1].split('_')[0]
                        data['policy_kwargs']['network'] = f"TransfromerNetwork{version}"
                    else:
                        data['policy_kwargs']['network'] = "CnnMlpNetwork2"
                elif "v3" in filename:
                    data['env_id'] = '2DBpp-v3'
                    data['total_timesteps'] = 12500000
                    data['env_kwargs']['min_items_per_bin'] = 20
                    data['env_kwargs']['max_items_per_bin'] = 30
                    if "transform" in filename:
                        version = filename.split('transform')[1].split('_')[0]
                        data['policy_kwargs']['network'] = f"TransfromerNetwork{version}"
                    else:
                        data['policy_kwargs']['network'] = "CnnMlpNetwork3"
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

            if "-F" in filename:
                data['PPO_kwargs']['add_invalid_probs'] = False
            else:
                data['PPO_kwargs']['add_invalid_probs'] = True

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
                elif version in ['3']:
                    data['policy_kwargs']['network_kwargs']['transformer_kwargs']['num_encoder_layers'] = num_layers
                    data['policy_kwargs']['network_kwargs']['transformer_kwargs']['num_decoder_layers'] = num_layers

            # update the YAML file
            with open(filepath, 'w') as file:
                yaml.dump(data, file, sort_keys=False)


def plot_training_curve_pe(
    ax: axes.Axes, 
    dirs: dict[str, str], 
    metric_name: str = 'ep_PE_mean',
    title: str = '10x10', 
    xlabel: str = 'Steps',
    ylabel: str = 'Avg Packing Efficiency',
    plot_moving_average: bool = True, 
    plot_raw: bool = True,
    window_size: int = 100,
) -> None:
    """
    Plot the training curve of packing efficiency from multiple metrics files, with option for raw or moving average.

    Args:
        ax: Matplotlib axis object to plot on.
        dirs (dict): Dictionary mapping labels to log paths.
        title (str): Title of the plot.
        use_moving_average (bool): If True, plot moving average; if False, plot raw values.
        window_size (int): Window size for moving average calculation.
    """
    colors = ['r', 'b', 'g', 'purple', 'orange', 'gray', 'magenta', 'olive', 'c', 'lime']  

    for idx, (label, log_file) in enumerate(dirs.items()):
        if not os.path.exists(log_file):
            print(f"Log file {log_file} does not exist.")
            continue

        df = pd.read_csv(log_file)
        df = df.dropna(subset=['timesteps', metric_name])
        df['timesteps'] = pd.to_numeric(df['timesteps'], errors='coerce')
        df[metric_name] = pd.to_numeric(df[metric_name], errors='coerce')
        df = df.dropna()

        timesteps = df['timesteps'].values
        ep_pe_mean = df[metric_name].values

        if plot_moving_average:
            ep_pe_mean_ma = np.convolve(ep_pe_mean, np.ones(window_size)/window_size, mode='valid')
            timesteps_ma = timesteps[window_size-1:]
            ax.plot(timesteps_ma, ep_pe_mean_ma, label=label, color=colors[idx], linewidth=1.5)        
        if plot_raw:
            ax.plot(timesteps, ep_pe_mean, color=colors[idx], linewidth=1.5, alpha=0.17)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def plot_training_curves_3subplots(
    dirs_dict: dict, 
    metric_name: str = 'ep_PE_mean',
    xlabel: str = 'Steps',
    ylabel: str = 'Avg Packing Efficiency',
    plot_moving_average: bool = True, 
    plot_raw: bool = True,
    window_size: int = 100,
    legend_loc: str = 'upper center',
    save_name: str = "training_curves.png",
) -> None:
    """
    Plot training curves for different grid sizes in a 2x2 subplot grid.

    Args:
        dirs_dict (dict): Dictionary mapping grid sizes to their dirs dictionaries.
        base_title (str): Base title for the figure.
        window_size (int): Window size for moving average calculation.
    """
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharex=False, sharey=False)
    axs = axs.flatten()  # Flatten the 1x3 array for easy iteration

    for ax, (bin_size, dirs) in zip(axs, dirs_dict.items()):
        if dirs:
            plot_training_curve_pe(
                ax, dirs, 
                metric_name=metric_name,
                title=str(bin_size),
                xlabel=xlabel,
                ylabel= ylabel,
                plot_moving_average=plot_moving_average, 
                plot_raw=plot_raw, 
                window_size=window_size,
            )
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc=legend_loc, bbox_to_anchor=(0.5, 1), ncols=len(list(dirs_dict.values())[0]), fontsize=12, edgecolor='#555')
    # plt.tight_layout(pad=1.15)
    plt.subplots_adjust(left=0.05, right=0.97, top=0.86, wspace=0.17)
    plt.savefig(save_name)
    plt.close()
    

def plot_compare_algo():
    compare_algo = {
        "10x10" : {
            "PaT":  "backup/main/2DBpp-v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T/metrics.csv",
            "Zhao-2D": "backup/compare/2DBpp-v1_zhao_ACKTR-h200-n64-M7-rA-P/metrics.csv",
            "Deep-Pack": "backup/compare/2DBpp-v1_deeppack_DDQN-h200-rC/metrics.csv",
            "Zhao-2D-truth": "backup/compare/2DBpp-v1_zhao_ACKTR-h200-n64-M7-rA-T/metrics.csv",
            "Deep-Pack-truth": "backup/compare/2DBpp-v1_deeppack_mask_DDQN-h200-rC/metrics.csv",
 
        }, 
        "20x20" : {
            "PaT":  "backup/main/2DBpp-v2_PPO-h400-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T/metrics.csv",
            "Zhao-2D": "backup/compare/2DBpp-v2_zhao_ACKTR-h400-n64-M7-rA-P/metrics.csv",
            "Deep-Pack": "backup/compare/2DBpp-v2_deeppack_DDQN-h400-rC/metrics.csv",
            "Zhao-2D-truth": "backup/compare/2DBpp-v2_zhao_ACKTR-h400-n64-M7-rA-T/metrics.csv",
            "Deep-Pack-truth": "backup/compare/2DBpp-v2_deeppack_mask_DDQN-h400-rC/metrics.csv",
        },
        "40x40" : {
            "PaT":  "backup/main/2DBpp-v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T/metrics.csv",
            "Zhao-2D": "backup/compare/2DBpp-v3_zhao_ACKTR-h1600-n64-M7-rA-P/metrics.csv",
            "Deep-Pack": "backup/compare/2DBpp-v3_deeppack_DDQN-h1600-rC/metrics.csv",
            "Zhao-2D-truth": "backup/compare/2DBpp-v3_zhao_ACKTR-h1600-n64-M7-rA-T/metrics.csv",
            "Deep-Pack-truth": "backup/compare/2DBpp-v3_deeppack_mask_DDQN-h1600-rC/metrics.csv",
        },
    }
    
    plot_training_curves_3subplots(
        compare_algo, 
        metric_name='ep_PE_mean',
        ylabel='Avg Packing Efficiency',
        window_size=100, 
        legend_loc='upper center',
        save_name="fig/Training Curves of Packing Efficiency for Different Algorithms.png"
    )
    
    plot_training_curves_3subplots(
        compare_algo, 
        metric_name='loss',
        ylabel='Avg Total Loss',
        window_size=100, 
        legend_loc='upper center',
        save_name="fig/Training Curves of Loss for Different Algorithms.png"
    )


def plot_cnn_net():
    compare_cnn_net = {
        "10x10" : {
            "PaT":  "backup/main/2DBpp-v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T/metrics.csv",
            "PaC": "backup/cnn_net/2DBpp-v1_PPO-h200-c02-n64-b32-R15-k1-rA/metrics.csv",
            # "Zhao-2D-truth": "backup/compare/2DBpp-v1_zhao_ACKTR-h200-n64-M7-rA-T/metrics.csv",
        }, 
        "20x20" : {
            "PaT":  "backup/main/2DBpp-v2_PPO-h400-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T/metrics.csv",
            "PaC": "backup/cnn_net/2DBpp-v2_PPO-h400-c02-n64-b32-R15-k1-rA/metrics.csv",
            # "Zhao-2D-truth": "backup/compare/2DBpp-v2_zhao_ACKTR-h400-n64-M7-rA-T/metrics.csv",
        },
        "40x40" : {
            "PaT":  "backup/main/2DBpp-v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T/metrics.csv",
            "PaC": "backup/cnn_net/2DBpp-v3_PPO-h1600-c02-n64-b32-R15-k1-rA/metrics.csv",
            # "Zhao-2D-truth": "backup/compare/2DBpp-v3_zhao_ACKTR-h1600-n64-M7-rA-T/metrics.csv",
        },
    }
    
    plot_training_curves_3subplots(
        compare_cnn_net, 
        metric_name='ep_PE_mean',
        ylabel='Avg Packing Efficiency',
        window_size=100, 
        legend_loc='upper center',
        save_name="fig/Training Curves of Packing Efficiency for using CNN.png"
    )
    
    plot_training_curves_3subplots(
        compare_cnn_net, 
        metric_name='loss',
        ylabel='Avg Total Loss',
        window_size=100, 
        legend_loc='upper center',
        save_name="fig/Training Curves of Loss for using CNN.png"
    )


def plot_compare_diff_reward_func():
    diff_reward_func = {
        "10x10" : {
            "PaT (area)":  "backup/main/2DBpp-v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T/metrics.csv",
            "PaT (cluster size x compactness)": "backup/diff_reward_type/2DBpp-v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rC-T/metrics.csv",
        }, 
        "20x20" : {
            "PaT (area)":  "backup/main/2DBpp-v2_PPO-h400-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T/metrics.csv",
            "PaT (cluster size x compactness)": "backup/diff_reward_type/2DBpp-v2_PPO-h400-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rC-T/metrics.csv",
        },
        "40x40" : {
            "PaT (area)":  "backup/main/2DBpp-v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T/metrics.csv",
            "PaT (cluster size x compactness)": "backup/diff_reward_type/2DBpp-v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rC-T/metrics.csv",
        },
    }
    
    plot_training_curves_3subplots(
        diff_reward_func, 
        metric_name='ep_PE_mean',
        ylabel='Avg Packing Efficiency',
        window_size=100, 
        legend_loc='upper center',
        save_name="fig/Training Curves of Packing Efficiency for Different Reward Functions.png"
    )
    
    plot_training_curves_3subplots(
        diff_reward_func, 
        metric_name='loss',
        ylabel='Avg Total Loss',
        window_size=100, 
        legend_loc='upper center',
        save_name="fig/Training Curves of Loss for Different Reward Functions.png"
    )


def plot_without_decoder():
    without_decoder = {
        "10x10" : {
            "PaT":  "backup/main/2DBpp-v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T/metrics.csv",
            "PaT-w/o decoder": "backup/without_decoder/2DBpp-v1_PPO-h200-c02-n64-b32-R15-transform1_TF,64,4,256,0,1-k1-rA-T/metrics.csv",
        }, 
        "20x20" : {
            "PaT":  "backup/main/2DBpp-v2_PPO-h400-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T/metrics.csv",
            "PaT-w/o decoder": "backup/without_decoder/2DBpp-v2_PPO-h400-c02-n64-b32-R15-transform1_TF,64,4,256,0,1-k1-rA-T/metrics.csv",
        },
        "40x40" : {
            "PaT":  "backup/main/2DBpp-v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T/metrics.csv",
            "PaT-w/o decoder": "backup/without_decoder/2DBpp-v3_PPO-h1600-c02-n64-b32-R15-transform1_TF,64,4,256,0,1-k1-rA-T/metrics.csv",
        },
    }
    
    plot_training_curves_3subplots(
        without_decoder, 
        metric_name='ep_PE_mean',
        ylabel='Avg Packing Efficiency',
        window_size=100, 
        legend_loc='lower right',
        save_name="fig/Training Curves of Packing Efficiency for Without Transformer Decoder.png"
    )
    
    plot_training_curves_3subplots(
        without_decoder, 
        metric_name='loss',
        ylabel='Avg Total Loss',
        window_size=100, 
        save_name="fig/Training Curves of Loss for Without Transformer Decoder.png"
    )




if __name__ == "__main__":
    # $ tensorboard --logdir backup
    if not os.path.exists("fig"):
        os.makedirs("fig")
    
    ''' Example usage of formatted_result '''
    # formatted_result(root_dir='backup')
    
    ''' Example usage of updating YAML files according to filename '''
    # update_yaml_files(root_dir='settings/main')

    ''' Plot training curves for different algorithms '''
    plot_compare_algo()

    ''' Plot training curves for using CNN network '''
    plot_cnn_net()
    
    ''' Plot training curves for different reward functions '''
    plot_compare_diff_reward_func()
    
    ''' Plot training curves for without decoder '''
    plot_without_decoder()
    