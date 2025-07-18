# PaT: Transformer-based Deep Reinforcement Learning for Large-sized Online 2D Bin Packing
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/JeepWay/PaT)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/JeepWay/PaT/blob/main/LICENSE)  

## 1. Environment
- OS: Ubuntu 20.04.1
- Python: 3.9.15
- PyTorch: 2.4.1+cu118
- stable-baselines3: 2.3.2
> Note: The training results might change across different hardware configurations (e.g., different GPUs), even under the same .yaml setting file and OS.


## 2. Installation
### Create new conda environment
```bash
# only cpu
bash scripts/install.sh -c 0

# use cuda (version: 11.8) (recommended)
bash scripts/install.sh -c 11.8
```
You may need to modify the path to your `conda.sh` file in the `install.sh` script, depending on where Miniconda or Anaconda is installed on your system.

### Use existing conda environment
```bash
# use cuda
pip install -r requirements.txt
```


## 3. Repository structure
```bash
├── envs/                  # 2D bin packing environment
│   ├── bpp/               # Core bin packing environment  
│   │   ├── __init__.py    # Package (bpp) setup
│   │   ├── bpp.py         # Main BppEnv class  
│   │   ├── bin.py         # Manage bin state
│   │   ├── creator.py     # Create requested items
│   │   └── item.py        # Item class definition  
│   └── register.py        # Environment registration 
├── pat/                   # DRL agent implementation (PPO with Transformer)  
│   ├── common/            # Custom Stable-Baselines3 (SB3) common utilities
│   ├── ppo/               # PPO algorithm implementation (SB3)
│   ├── __init__.py        # Package (pat) setup
├── save_weight/           # Trained model weights of main method
│   ├── 2DBpp-v1_PPO-*/    # 10x10 environment 
│   ├── 2DBpp-v2_PPO-*/    # 20x20 environment 
│   └── 2DBpp-v3_PPO-*/    # 40x40 environment
├── scripts/               # Installation and execution scripts  
│   ├── install.sh         # Environment setup (CPU/GPU)  
│   ├── all.sh             # Commands to run all experiments  
│   ├── run_docker_cpu.sh  # Docker CPU deployment  
│   └── run_docker_gpu.sh  # Docker GPU deployment  
├── settings/              # YAML configuration files for experiments  
│   ├── cnn_net/           # CNN-based network
│   ├── diff_reward_type/  # Different reward function
│   ├── general/           # Generalization mechanism testing
│   ├── hybrid_net/        # Hybrid network (CNN + Transformer)
│   ├── main/              # Main methods
│   ├── mixed/             # Mixed-training
│   ├── mask_replace/      # Grid search of replace method in action masking
│   └── multi_layer/       # Multi-layer for Transformer  
├── test_data/             # Requested item sequences in testing phase
│   ├── 2DBpp-v1/          # 10x10 environment 
│   ├── 2DBpp-v2/          # 20x20 environment 
│   └── 2DBpp-v3/          # 40x40 environment
├── get_test_data.py       # Get the information of requested item sequences in testing
├── main.py                # Main entry point for single-env training/testing  
├── main_mixed.py          # Mixed environment training  
├── main_general.py        # Generalization mechanism testing
├── UI.py                  # Gradio-based visualization interface to demo bin packing progress
└── utils.py               # Functions to format result, update .yaml files, and plot training curves.
```

## 4. Usage example
### 4.1 Run the main methods with configuration files
All experiment configuration files (.yaml) are organized under the [/settings](/settings/) directory, categorized by different network types (e.g., main, hybrid_net, multi_layer, cnn_net, etc.).

```bash
# 10x10
python main.py --config_path settings/main/v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml

# 20x20
python main.py --config_path settings/main/v2_PPO-h400-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml

# 40x40
python main.py --config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
```

For more examples, please refer to the [scripts/all.sh](scripts/all.sh) script, which contains the complete commands for all experiments in the thesis. 

Each command corresponds to specific YAML configuration files in the settings/ directory, ensuring experimental reproducibility and systematic management.

For more details about the configuration files, please refer to the [update_yaml_files](utils.py#L69) function in the utils.py file.

### 4.2 Choose training, testing mode, or both
You can choose to run the training, testing, or both modes by adding the `--mode` flag into the command, you can also set it to `train` or `test`, or `both`. If you don't set the `--mode` flag, it will default to `both` mode.

For example, if you want to run the testing mode, you can execute the following command:

```bash
python main.py --config_path settings/main/v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml --mode test
```

### 4.3 Watch the training process in tensorboard (Optional)
if you want to watch the training process in tensorboard, you can execute the following command in the terminal:
```bash
tensorboard --logdir=logs/
```

### 4.4 Upload the training process to wandb (Optional)
If you want to upload the training results to Weights & Biases (wandb), you need to add the `--use_wandb` flag to the command. For example:

```bash
# 10x10
python main.py --config_path settings/main/v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml --use_wandb
```

After executing the above command, you will be asked to choose the visualization mode like below.

```bash
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 
```
If you want to visualize the training results on the wandb website, you can choose the second option, and then you will be asked to paste the API key of your wandb account. 

If you don't have one, you can choose the first option to create a new account.

If you just want to save the training results locally, you can choose the third option.

## 4.5 Watch the bin packing process (Optional)
If you want to watch the progress of the bin packing process, you can run `UI.py` file, which provides a Gradio-based interface to visualize the bin packing process.

```bash
python UI.py
```


## 5. Use docker images (Optional)
### Build docker image
Build CPU image:
```bash
make docker-cpu
```
Build GPU image (with nvidia-docker):
```bash
make docker-gpu
```

### Run GPU images
Run the nvidia-container-toolkit GPU image
```bash
docker run -it --rm --gpus=all --volume $(pwd):/home/user/pat jeepway/pat-gpu:latest bash -c "cd /home/user/pat && ls && pwd && /bin/bash"
```

Or, use `make` command to run with the shell file
```bash
make docker-run-gpu
```
After executing the above command, you will enter the terminal of the container.

### Run CPU images
Run the docker CPU image
```bash
docker run -it --rm --volume $(pwd):/home/user/pat jeepway/pat-cpu:latest bash -c "cd /home/user/pat && ls && pwd && /bin/bash"
```

Or, use `make` command to run with the shell file
```bash
make docker-run-cpu
```
After executing the above command, you will enter the terminal of the container.


## 6. Compared algorithms
you can watch the source code of the compared algorithms in this [repository](https://github.com/JeepWay/pat-compare-algorithm)
