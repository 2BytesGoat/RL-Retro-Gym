import argparse
import json
import retro
import torch
import tqdm
import time
import numpy as np
import cv2

from pathlib import Path

from utils import yaml_config_hook
from custom_monitor import CustomMonitor
from src.processing import get_transforms
from src.modeling.autoencoders import get_encoder
from src.modeling.agents import DQN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", '-s', default="./checkpoints", type=Path)
    parser.add_argument("--load_path", '-l', type=Path, help="Path where pretrained encoder and agent weights are stored")
    parser.add_argument("--enc_config_path", '-ec', required=True, type=Path)
    parser.add_argument("--agent_config_path", '-ac', required=True, type=Path)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    enc_config = yaml_config_hook(args.enc_config_path)
    agent_config = yaml_config_hook(args.agent_config_path)

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===================Intialize environment=====================
    env = retro.make(game='SuperMarioKart-Snes', state='1P_DK_Shroom_R1_fast')
    init_state = env.reset()

    simplified_actions = agent_config['simplified_actions']

    transforms = get_transforms(**enc_config['preprocessing'])
    init_state = apply_transforms(init_state, transforms)
    
    input_shape = init_state.shape[1:]
    action_shape = len(simplified_actions)

    # ===================Initialize monitor=====================
    h, w, c = init_state.shape
    monitor = CustomMonitor(args.save_path, (w, h))

    # ===================Initialize encoder=====================
    encoder = get_encoder(input_shape=input_shape, **enc_config['encoder']).to(DEVICE)

    if args.load_path and args.load_path.exists():
        encoder.load_state_dict(torch.load(args.load_path / "best_encoder.pt"))

    # ===================Initialize agent=====================
    agent = DQN(action_shape, 
                encoder=encoder,
                batch_size=agent_config['training']['batch_size'],
                load_path=args.load_path,
                device=DEVICE)

    # ===================Train agent=====================
    n_episodes = agent_config['training']['n_episodes']
    n_steps = agent_config['training']['n_steps']
    prev_best_reward = 0
    for i_episode in range(n_episodes):
        # ===================Visualize agent=====================
        if i_episode % 5 == 0:
            frame = env.reset()
            state = apply_transforms(frame, transforms)

            monitor.start_new(name=f'episode_{i_episode}')
            monitor.write(frame, state=state)
            for _ in range(n_steps):
                # Select and perform an action
                action = agent.take_action(state, greedy=True)
                # Format action for environment
                env_action = simplified_actions[action.item()]
                # Apply action on environment
                frame, reward, done, info = env.step(env_action)
                state = apply_transforms(frame, transforms)
                # Record next state
                monitor.write(frame, state=state)
                env.render()
            monitor.release()

        # Initialize the environment and state
        ep_reward = []
        prev_cart_pos = 0
        frame = env.reset()
        state = apply_transforms(frame, transforms)
        pbar = tqdm.tqdm(total=n_steps, position=0)
        for i in range(n_steps):
            # Select and perform an action
            action = agent.take_action(state)

            # Format action for environment
            env_action = simplified_actions[action.item()]

            # Take action in environment
            n_frame, _, done, info = env.step(env_action)
            reward = calculate_reward(info, prev_cart_pos)
            prev_cart_pos = info['cart_position_x']

            ep_reward.append(reward)

            # Store the transition in memory
            n_state = apply_transforms(n_frame, transforms)
            agent.memory.push(state, action, n_state, torch.tensor([reward]))

            # Move to the next state
            state = n_state

            # Perform one step of the optimization (on the target network)
            agent.optimize_agent()
            pbar.update(1)

            if done:
                break

            last_return = round(reward, 2)
            mean_ep_reward = round(np.mean(ep_reward), 4)
            pbar.set_description_str(f"Episode: {i_episode} | Steps: {i+1} | Last reward: {last_return} | Average reward: {mean_ep_reward}")
        pbar.close()
        # Update the target network, copying all weights and biases in DQN
        if i_episode % 10 == 0:
            prev_best_reward = mean_ep_reward
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            torch.save(agent.target_net.state_dict(), save_path / 'best_agent.pt')

    env.close()

def calculate_reward(info, prev_pos):
    if prev_pos < info['cart_position_x']:
        advance_reward = 10
    else:
        advance_reward = 0

    cart_place = 9.0 - (info['cart_place'] / 2 + 1)
    cart_speed = info['cart_speed'] / 80
    return cart_place * advance_reward + cart_speed

def apply_transforms(state, transforms):
    _state = state.copy()
    for t in transforms:
        _state = t(_state)
    return _state

if __name__ == '__main__':
    main()