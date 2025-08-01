import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from envs.register import registration_envs
import pat
from pat import PPO
from pat.common.constants import BIN
import sys
sys.modules['mask_pack'] = pat
import os
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.makedirs("fig", exist_ok=True)
os.makedirs("fig/10x10", exist_ok=True)
os.makedirs("fig/20x20", exist_ok=True)
os.makedirs("fig/40x40", exist_ok=True)
registration_envs()
seed = 10

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-5s (%(filename)s:%(lineno)d): %(message)s",
    datefmt="%H:%M:%S",
)

env_kwargs_10x10 = {
    "render_mode": "human",
    "bin_channels": 3,
    "min_items_per_bin": 10,
    "max_items_per_bin": 20,
    "area_reward_coef": 0.4,
    "constant_penalty": -5.0,
    "action_fail": "continue",
    "reward_type": "area"
}

env_kwargs_20x20 = {
    "render_mode": "human",
    "bin_channels": 3,
    "min_items_per_bin": 15,
    "max_items_per_bin": 25,
    "area_reward_coef": 0.4,
    "constant_penalty": -5.0,
    "action_fail": "continue",
    "reward_type": "area"
}

env_kwargs_40x40 = {
    "render_mode": "human",
    "bin_channels": 3,
    "min_items_per_bin": 20,
    "max_items_per_bin": 30,
    "area_reward_coef": 0.4,
    "constant_penalty": -5.0,
    "action_fail": "continue",
    "reward_type": "area"
}


# 可以把以下權重路徑替換成新的其他權重路徑
args = [
    {
        "label": "10x10",
        "weight_path": "save_weight/2DBpp-v1_PPO-h200-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T/2DBpp-v1.zip",
        "env_id": "2DBpp-v1",
        "env_kwargs": env_kwargs_10x10,
    },
    {
        "label": "20x20",
        "weight_path": "save_weight/2DBpp-v2_PPO-h400-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T/2DBpp-v2.zip",
        "env_id": "2DBpp-v2",
        "env_kwargs": env_kwargs_20x20,
    },
    {
        "label": "40x40",
        "weight_path": "save_weight/2DBpp-v3_PPO-h1600-c02-n64-b32-R15-transform3_TF,64,4,256,0,1-k1-rA-T/2DBpp-v3.zip",
        "env_id": "2DBpp-v3",
        "env_kwargs": env_kwargs_40x40,
    },
]

PPO_kwargs = {
    "policy": "CnnMlpPolicy",
    "clip_range": 0.2,
    "learning_rate": 0.001,
}


def step(env, action_width, action_height, cur_steps):
    if (type(action_width) is str) or (type(action_height) is str):
        action_width = int(action_width)
        action_height = int(action_height)
    action = env.unwrapped.bin.location_to_index(action_width, action_height)
    new_observations, reward, terminated, truncated, info = env.step(action)
    state = new_observations[BIN][0]
    item_width = info["next_item"][0]
    item_height = info["next_item"][1]
    step_btn = gr.Button("Step", visible=True)
    info_box = gr.Markdown("## Packing information", visible=False)
    cur_steps += 1
    
    if terminated or truncated:
        step_btn = gr.Button("Step", visible=False)
        info_box = gr.Markdown(f"## Packing information \
                                        \n### You have used up all step of the current episode. \
                                        \n### The space utilization of this episode is  {info['SU']*100}%. \
                                        \n### Please click the reset button to start the new episode.", visible=True)
    return new_observations, state, item_width, item_height, step_btn, info_box, cur_steps


def reset(env):
    observations, _ = env.reset()
    logging.info(f"items_per_bin: {env.unwrapped.items_creator.items_per_bin}, "
                 f"items_list: {env.unwrapped.items_creator.items_list}")
    state = observations[BIN][0]
    item_width = observations[BIN][1][0][0]
    item_height = observations[BIN][2][0][0]
    action_width = 0
    action_height = 0
    step_btn = gr.Button("Step", visible=True)
    info_box = gr.Markdown("## Packing information", visible=False)
    cur_steps = 0
    plt.close()
    return observations, state, item_width, item_height, action_width, action_height, step_btn, info_box, cur_steps


def render(state, cur_steps:int, action_width=None, action_height=None):
    if (type(action_width) is str) or (type(action_height) is str):
        action_width = int(action_width)
        action_height = int(action_height)
    state_ = state
    bin_w, bin_h = state_.shape
    state_ = np.rot90(state_, k=1)
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, bin_w)
    ax.set_ylim(0, bin_h)
    if bin_w >= 32:
        ax.set_xticks(range(bin_w+1), range(bin_w+1), size=6)
        ax.set_yticks(range(bin_h+1), range(bin_h+1), size=6)
    else:
        ax.set_xticks(range(bin_w+1), range(bin_w+1))
        ax.set_yticks(range(bin_h+1), range(bin_h+1))   
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.grid(True)

    # Draw the grid
    for y in range(bin_h):
        for x in range(bin_w):
            if state_[y][x] == 0:
                ax.add_patch(plt.Rectangle((x, bin_h - y - 1), 1, 1, color="black"))
            else:
                ax.add_patch(plt.Rectangle((x, bin_h - y - 1), 1, 1, edgecolor="gray", fill=None))

    # Mark the item
    if (action_width is not None) and (action_height is not None):
        ax.scatter(action_width, action_height, s=50, color='red', marker='o')
    ax.set_title(f"{bin_w}x{bin_w} bin")
    fig.savefig(f"fig/{bin_w}x{bin_h}/bin_state_{bin_w}x{bin_h}_step{cur_steps}.png")
    return fig


def update_agent_choice(agent, env, observations):
    action, _ = agent.policy.predict(observations, deterministic=True)
    action_width, action_height = env.unwrapped.bin.index_to_location(int(action))
    return action_width, action_height


def get_total_steps(env):
    return env.unwrapped.items_creator.items_per_bin
        

with gr.Blocks() as demo:
    gr.HTML("<h1 style='text-align: center;'>2D Bin Packing Visualization</h1>")
    for arg in args:
        with gr.Tab(label=arg["label"]):
            gr.Markdown("# AI Agent Section")
            with gr.Row() as ai_agent_section:    
                agent = PPO.load(arg["weight_path"], **PPO_kwargs)
                env = gym.make(arg["env_id"], **arg["env_kwargs"])
                observations, _ = env.reset(seed=seed)
                bin_w, bin_h = observations[BIN][0].shape
                item_w = observations[BIN][1][0][0]
                item_h = observations[BIN][2][0][0]

                agent = gr.State(agent)
                env = gr.State(env)
                state = gr.State(observations[BIN][0])
                observations = gr.State(observations)
                
                with gr.Column(scale=1, min_width=600):
                    with gr.Row():
                        cur_steps = gr.Number(0, label="Current steps", interactive=False)
                        total_steps = gr.Number(0, label="Total steps", interactive=False)
                    with gr.Row():
                        item_width = gr.Number(value=item_w, label="Next item's width", interactive=False)
                        item_height = gr.Number(value=item_h, label="Next item's height", interactive=False)  
                    with gr.Row():
                        action_width = gr.Number(value=0, label="Agent's choice for place index of width", interactive=False)
                        action_height = gr.Number(value=0, label="Agent's choice for place index of height", interactive=False)
                    with gr.Row():
                        step_btn = gr.Button("Step")
                        reset_btn = gr.Button("Reset")
                    info_box = gr.Markdown("## Packing information", visible=False)
                with gr.Column(scale=2, min_width=600): 
                    bin_plot = gr.Plot(label="Bin state")
                    
                demo.load(
                    render, 
                    inputs=[state, cur_steps], 
                    outputs=[bin_plot],
                ).then(
                    update_agent_choice,
                    inputs=[agent, env, observations],
                    outputs=[action_width, action_height],
                ).then(
                    get_total_steps,
                    inputs=[env],
                    outputs=[total_steps],
                )
                
                step_btn.click(
                    step, 
                    inputs=[env, action_width, action_height, cur_steps], 
                    outputs=[observations, state, item_width, item_height, step_btn, info_box, cur_steps],
                ).then(
                    render, 
                    inputs=[state, cur_steps, action_width, action_height], 
                    outputs=[bin_plot],
                ).then(
                    update_agent_choice,
                    inputs=[agent, env, observations],
                    outputs=[action_width, action_height],
                )
                
                reset_btn.click(
                    reset, 
                    inputs=[env],
                    outputs=[observations, state, item_width, item_height, action_width, action_height, step_btn, info_box, cur_steps],
                ).then(
                    render, 
                    inputs=[state, cur_steps], 
                    outputs=[bin_plot],
                ).then(
                    get_total_steps,
                    inputs=[env],
                    outputs=[total_steps],
                )

            gr.HTML("<hr style='color:#d0d0d5;border-width:2.5px'>")
            gr.Markdown("# User Section")
            with gr.Row() as user_section:
                env = gym.make(arg["env_id"], **arg["env_kwargs"])
                observations, _ = env.reset(seed=seed)
                bin_w, bin_h = observations[BIN][0].shape
                item_w = observations[BIN][1][0][0]
                item_h = observations[BIN][2][0][0]
                
                env = gr.State(env)
                state = gr.State(observations[BIN][0])
                observations = gr.State(observations)
                
                with gr.Column(scale=1, min_width=600):
                    with gr.Row():
                        cur_steps = gr.Number(0, label="Current steps", interactive=False)
                        total_steps = gr.Number(0, label="Total steps", interactive=False)
                    with gr.Row():
                        item_width = gr.Number(value=item_w, label="Next item's width", interactive=False)
                        item_height = gr.Number(value=item_h, label="Next item's height", interactive=False)  
                    with gr.Row():
                        action_width = gr.Slider(0, bin_w-1, value=0, step=1, label="Select place index of width")
                        action_height = gr.Slider(0, bin_h-1, value=0, step=1, label="Select place index of height")
                    with gr.Row():
                        step_btn = gr.Button("Step")
                        reset_btn = gr.Button("Reset")
                    info_box = gr.Markdown("## Packing information", visible=False)
                with gr.Column(scale=2, min_width=600): 
                    bin_plot = gr.Plot(label="Bin state")
        
                demo.load(
                    render, 
                    inputs=[state, cur_steps], 
                    outputs=[bin_plot],
                ).then(
                    get_total_steps,
                    inputs=[env],
                    outputs=[total_steps],
                )
                
                step_btn.click(
                    step, 
                    inputs=[env, action_width, action_height, cur_steps], 
                    outputs=[observations, state, item_width, item_height, step_btn, info_box, cur_steps],
                ).then(
                    render, 
                    inputs=[state, cur_steps, action_width, action_height], 
                    outputs=[bin_plot],
                )
                
                reset_btn.click(
                    reset, 
                    inputs=[env],
                    outputs=[observations, state, item_width, item_height, action_width, action_height, step_btn, info_box, cur_steps],
                ).then(
                    render, 
                    inputs=[state, cur_steps],  
                    outputs=[bin_plot],
                ).then(
                    get_total_steps,
                    inputs=[env],
                    outputs=[total_steps],
                )
            
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="2D Bin Packing Environment")
    parser.add_argument('--share', default=True, help="Whether to share the Gradio app publicly. Default is True.")
    args = parser.parse_args()
    demo.launch(share=args.share)
