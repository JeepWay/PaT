# main setting
python main.py --config_path settings/transformer3/v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/transformer3/v2_PPO-h400-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/transformer3/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/transformer3/v4_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T.yaml


# ablation/orig_net
python main.py --config_path settings/ablation/orig_net/v1_PPO-h200-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/ablation/orig_net/v2_PPO-h400-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/ablation/orig_net/v3_PPO-h1600-c02-n64-b32-R15-k1-rA.yaml
python main.py --config_path settings/ablation/orig_net/v4_PPO-h1600-c02-n64-b32-R15-k1-rA.yaml


# ablation/only_encoder
python main.py --config_path settings/ablation/only_encoder/v1_PPO-h200-c02-n64-b32-R15-transform1_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/ablation/only_encoder/v2_PPO-h400-c02-n64-b32-R15-transform1_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/ablation/only_encoder/v3_PPO-h1600-c02-n64-b32-R15-transform1_TF,64,4,256,0,1-k1-rA-T.yaml
python main.py --config_path settings/ablation/only_encoder/v4_PPO-h1600-c02-n64-b32-R15-transform1_TF,64,4,256,0,1-k1-rA-T.yaml


# ablation/pred_mask
python main.py --config_path settings/transformer3/v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-P.yaml
python main.py --config_path settings/transformer3/v2_PPO-h400-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-P.yaml
python main.py --config_path settings/transformer3/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-P.yaml
python main.py --config_path settings/transformer3/v4_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-P.yaml


# finetune
python main_mixed.py --config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T-mixed.yaml
python finetune.py --config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T-mixed.yaml --finetune_config settings/finetune/v1_10000.yaml
python finetune.py --config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T-mixed.yaml --finetune_config settings/finetune/v2_10000.yaml
python finetune.py --config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T-mixed.yaml --finetune_config settings/finetune/v3_10000.yaml
python finetune.py --config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T-mixed.yaml --finetune_config settings/finetune/v5_10000.yaml
python finetune.py --config_path settings/main/v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T-mixed.yaml --finetune_config settings/finetune/v6_10000.yaml
