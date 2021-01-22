import argparse
import json
import retro
import torch
import tqdm
import time
import numpy as np

from pathlib import Path

from src.processing import get_transforms
from src.modeling.agents import DQN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default=None, type=Path)
    parser.add_argument("--load_path", required=True, type=Path)

    parser.add_argument("--encoder_type", required=True, type=str)
    parser.add_argument("--config_name", required=True)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = load_json('./configs', args.config_name)
    config = config[args.encoder_type]

    simplified_actions = {
        0: 6, # left
        1: 7, # right
        2: 8, # decelerate
        3: 0  # accelerate
    }

    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    # ===================Intialize environment=====================
    env = retro.make(game='SuperMarioKart-Snes')
    init_state = env.reset()

    # ===================Initialize agent=====================
    transforms = get_transforms(roi=config['roi'], grayscale=config['grayscale'])
    
    init_state = apply_transforms(init_state, transforms)
    
    state_shape = init_state.shape[1:]
    # action_shape = env.action_space.shape[0]
    action_shape = len(simplified_actions)

    agent = DQN(state_shape, action_shape, load_pretrained=args.load_path)

    state = env.reset()
    state = apply_transforms(state, transforms)

    # ===================Train agent=====================
    num_episodes = 500
    num_ep_steps = 1000
    prev_best_reward = 0
    for i_episode in range(num_episodes):
        # ===================Visualize agent=====================
        if i_episode % 10 == 0:
            for _ in range(num_ep_steps):
                # Select and perform an action
                action = agent.take_action(state)
                # Format action for environment
                btn_pressed = simplified_actions[action.item()]
                env_action = [0] * len(env.buttons)
                env_action[btn_pressed] = 1
                n_state, reward, done, info = env.step(env_action)
                env.render()

        # Initialize the environment and state
        ep_reward = []
        state = env.reset()
        state = apply_transforms(state, transforms)
        prev_cart_pos = 0
        with tqdm.tqdm(total=num_ep_steps, position=0, leave=True) as pbar:
            for i in range(num_ep_steps):
                # Select and perform an action
                action = agent.take_action(state)

                # Format action for environment
                btn_pressed = simplified_actions[action.item()]
                env_action = [0] * len(env.buttons)
                env_action[btn_pressed] = 1

                # Take action in environment
                n_state, _, done, info = env.step(env_action)
                reward = calculate_reward(info, prev_cart_pos)
                prev_cart_pos = info['cart_position_x']

                ep_reward.append(reward)

                # Store the transition in memory
                n_state = apply_transforms(n_state, transforms)
                reward = torch.tensor([reward])
                agent.memory.push(state, action, n_state, reward)

                # Move to the next state
                state = n_state

                # Perform one step of the optimization (on the target network)
                agent.optimize_agent()
                pbar.update(1)

                if done:
                    break
                
            mean_ep_reward = round(np.mean(ep_reward), 4)
            print(f"\nEpisode: {i_episode} | Steps: {i} | Average return: {mean_ep_reward} | Max return: {max(ep_reward)} | Last return: {ep_reward[-1]}")

        # Update the target network, copying all weights and biases in DQN
        if i_episode % 10 == 0:
            prev_best_reward = mean_ep_reward
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            torch.save(agent.target_net.state_dict(), save_path / 'best_agent.pt')

    env.close()

def calculate_reward(info, prev_pos):
    if prev_pos < info['cart_position_x']:
        advance_reward = 1
    else:
        advance_reward = 0

    cart_place = 9.0 - (info['cart_place'] / 2 + 1)
    cart_speed = info['cart_speed'] - 10.0
    return advance_reward * cart_place + cart_speed * 1e-3

def apply_transforms(state, transforms):
    _state = state.copy()
    for t in transforms:
        _state = t(_state)
    return _state

def load_json(fdir, name):
    """
    Load json as python object
    """
    path = Path(fdir) / "{}.json".format(name)
    if not path.exists():
        raise FileNotFoundError("Could not find json file: {}".format(path))
    with open(path, "r") as f:
        obj = json.load(f)
    return obj

if __name__ == '__main__':
    main()