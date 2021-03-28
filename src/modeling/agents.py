"""
Start code: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from collections import namedtuple

from . import backbone

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# TODO: make this a dictionary
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.2
EPS_DECAY = 2000

class DQN:
    def __init__(self, action_shape, encoder, memory_len=10000, batch_size=128, load_path=None, ckpt_dst='', device=None):
        self.action_shape = action_shape
        self.encoder = encoder
        self.enc_dim = encoder.latent_dim
        self.memory_len = memory_len
        self.batch_size = batch_size
        self.load_path = load_path
        self.ckpt_dst = Path(ckpt_dst)
        self.device = device

        # ===================Initialize DQN backbone=====================
        self.target_net = backbone.MLP(self.enc_dim, self.action_shape).to(self.device)
        self.policy_net = backbone.MLP(self.enc_dim, self.action_shape).to(self.device)
        self.bb_optimizer = torch.optim.RMSprop(self.policy_net.parameters())

        if self.load_path and self.load_path.exists():
            if (self.load_path / "best_agent.pt").is_file():
                self.target_net.load_state_dict(torch.load(self.load_path / "best_agent.pt"))
                self.policy_net.load_state_dict(torch.load(self.load_path / "best_agent.pt"))

        # ===================Initialize replay buffer=====================
        self.memory = ReplayBuffer(memory_len)
        self.current_step = 0

    def take_action(self, state, greedy=False):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * self.current_step / EPS_DECAY)
        self.current_step += 1
        if sample > eps_threshold and not greedy:
            state = state.to(self.device)
            with torch.no_grad():
                # encode states
                enc_state = self.encoder.encode(state)
            # take the action with the maximum return
            return torch.argmax(self.policy_net(enc_state)).view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_shape)]], device=self.device, dtype=torch.long)

    def optimize_agent(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None]).to(self.device)
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        with torch.no_grad():
            enc_state_batch = self.encoder.encode(state_batch).to(self.device)
        state_action_values = self.policy_net(enc_state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            enc_n_state_batch = self.encoder.encode(non_final_next_states).to(self.device)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(enc_n_state_batch).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.bb_optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.bb_optimizer.step()


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)