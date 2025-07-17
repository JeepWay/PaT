# get testing data
python get_env_data.py --config_path settings/main/v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python get_env_data.py --config_path settings/main/v2_PPO-h400-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python get_env_data.py --config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml


# Main setting (PaT)
python main.py --config_path settings/main/v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/main/v2_PPO-h400-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml


# Replacing Transformer with CNN (PaC)
python main.py --config_path settings/cnn_net/v1_PPO-h200-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/cnn_net/v2_PPO-h400-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/cnn_net/v3_PPO-h1600-c02-n64-b32-R15-k1-rA.yaml


# Different reward functions (cluster size X compactness)
python main.py --config_path settings/diff_reward_type/v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rC-T.yaml
python main.py --config_path settings/diff_reward_type/v2_PPO-h400-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rC-T.yaml
python main.py --config_path settings/diff_reward_type/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rC-T.yaml


# Generalization Mechanism (direct transfer and testing)
python main_general.py --orig_config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml --target_config_path settings/general/v1_10000.yaml
python main_general.py --orig_config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml --target_config_path settings/general/v2_10000.yaml
python main_general.py --orig_config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml --target_config_path settings/general/v3_10000.yaml
python main_general.py --orig_config_path settings/main/v2_PPO-h400-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml --target_config_path settings/general/v1_10000.yaml


# Generalization Mechanism (mixed training and testing)
python main_mixed.py --config_path settings/mixed/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T-mixed-111.yaml    # method 2
python main_mixed.py --config_path settings/mixed/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T-mixed-311.yaml    # method 3
python main_general.py --orig_config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T-mixed-111.yaml --target_config_path settings/general/v1_10000.yaml
python main_general.py --orig_config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T-mixed-111.yaml --target_config_path settings/general/v2_10000.yaml
python main_general.py --orig_config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T-mixed-111.yaml --target_config_path settings/general/v3_10000.yaml


# Multi layer for Transformer (2 layers)
python main.py --config_path settings/multi_layer/v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,2-k1-rA-T.yaml
python main.py --config_path settings/multi_layer/v2_PPO-h400-c02-n64-b32-R15-transform3_TF,64,4,256,0,2-k1-rA-T.yaml
python main.py --config_path settings/multi_layer/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,2-k1-rA-T.yaml


# Grid search of the coefficient for replace mask (10x10)
python main.py --config_path settings/mask_replace/v1_PPO-h200-c02-n64-b32-R7-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v1_PPO-h200-c02-n64-b32-R30-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v1_PPO-h200-c02-n64-b32-R50-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v1_PPO-h200-c02-n64-b32-R100-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v1_PPO-h200-c02-n64-b32-Re3-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v1_PPO-h200-c02-n64-b32-Re4-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v1_PPO-h200-c02-n64-b32-Re5-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v1_PPO-h200-c02-n64-b32-Re6-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v1_PPO-h200-c02-n64-b32-Re7-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v1_PPO-h200-c02-n64-b32-Re8-transform3_TF,64,4,256,0,1-k1-rA-T.yaml


# Grid search of the coefficient for replace mask (20x20)
python main.py --config_path settings/mask_replace/v2_PPO-h400-c02-n64-b32-R7-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v2_PPO-h400-c02-n64-b32-R30-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v2_PPO-h400-c02-n64-b32-R50-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v2_PPO-h400-c02-n64-b32-R100-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v2_PPO-h400-c02-n64-b32-Re3-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v2_PPO-h400-c02-n64-b32-Re4-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v2_PPO-h400-c02-n64-b32-Re5-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v2_PPO-h400-c02-n64-b32-Re6-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v2_PPO-h400-c02-n64-b32-Re7-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v2_PPO-h400-c02-n64-b32-Re8-transform3_TF,64,4,256,0,1-k1-rA-T.yaml


# Grid search of the coefficient for replace mask (40x40)
python main.py --config_path settings/mask_replace/v3_PPO-h1600-c02-n64-b32-R7-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v3_PPO-h1600-c02-n64-b32-R30-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v3_PPO-h1600-c02-n64-b32-R50-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v3_PPO-h1600-c02-n64-b32-R100-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v3_PPO-h1600-c02-n64-b32-Re3-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v3_PPO-h1600-c02-n64-b32-Re4-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v3_PPO-h1600-c02-n64-b32-Re5-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v3_PPO-h1600-c02-n64-b32-Re6-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v3_PPO-h1600-c02-n64-b32-Re7-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/mask_replace/v3_PPO-h1600-c02-n64-b32-Re8-transform3_TF,64,4,256,0,1-k1-rA-T.yaml


