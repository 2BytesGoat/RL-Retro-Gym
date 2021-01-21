import argparse
import json
import retro
import torch
import tqdm

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

    # ===================Intialize environment=====================
    env = retro.make(game='SuperMarioKart-Snes')
    init_state = env.reset()

    # ===================Initialize agent=====================
    transforms = get_transforms(roi=config['roi'], grayscale=config['grayscale'])
    
    init_state = apply_transforms(init_state, transforms)
    
    state_shape = init_state.shape[1:]
    action_shape = env.action_space.shape[0]

    agent = DQN(state_shape, action_shape, enc_load=args.load_path)

    # ===================Train agent=====================
    num_episodes = 50
    rewards = []
    for i_episode in tqdm.tqdm(range(num_episodes)):
        # Initialize the environment and state
        state = env.reset()
        state = apply_transforms(state, transforms)
        done = False
        ep_reward = []
        for i in tqdm.tqdm(range(1000)):
            # Select and perform an action
            action = agent.take_action(state)
            # Format action for environment
            env_action = [i == action.item() for i in range(len(env.buttons))]
            n_state, reward, done, info = env.step(env_action)
            ep_reward.append(reward)

            # Store the transition in memory
            n_state = apply_transforms(n_state, transforms)
            reward = torch.tensor([reward])
            agent.memory.push(state, action, n_state, reward)

            # Move to the next state
            state = n_state

            # Perform one step of the optimization (on the target network)
            agent.optimize_agent()

        rewards.append(ep_reward)

        # Update the target network, copying all weights and biases in DQN
        if i_episode % 10 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    done = False
    while not done:
        # Select and perform an action
        action = agent.take_action(state)
        # Format action for environment
        env_action = [i == action.item() for i in range(len(env.buttons))]
        n_state, reward, done, info = env.step(env_action)

        env.render()

    env.close()

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