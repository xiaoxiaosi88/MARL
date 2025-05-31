import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutBuffer:
    """MAPPO经验回放缓冲区"""

    def __init__(self, num_steps: int, num_agents: int, obs_dim: int,
                 action_dim: int, state_dim: Optional[int] = None,
                 use_centralized_v: bool = True, use_gae: bool = True,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 use_proper_time_limits: bool = False):

        self.num_steps = num_steps
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim if state_dim else obs_dim
        self.use_centralized_v = use_centralized_v
        self.use_gae = use_gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits

        # 初始化缓冲区
        self.reset()

    def reset(self):
        """重置缓冲区"""
        self.observations = torch.zeros(self.num_steps + 1, self.num_agents, self.obs_dim)
        self.states = torch.zeros(self.num_steps + 1, self.num_agents, self.state_dim)
        self.actions = torch.zeros(self.num_steps, self.num_agents, self.action_dim)
        self.rewards = torch.zeros(self.num_steps, self.num_agents, 1)
        self.value_preds = torch.zeros(self.num_steps + 1, self.num_agents, 1)
        self.returns = torch.zeros(self.num_steps + 1, self.num_agents, 1)
        self.action_log_probs = torch.zeros(self.num_steps, self.num_agents, 1)
        self.advantages = torch.zeros(self.num_steps, self.num_agents, 1)
        self.masks = torch.ones(self.num_steps + 1, self.num_agents, 1)
        self.bad_masks = torch.ones(self.num_steps + 1, self.num_agents, 1)
        self.active_masks = torch.ones(self.num_steps + 1, self.num_agents, 1)

        self.step = 0

    def insert(self, obs: torch.Tensor, states: torch.Tensor, actions: torch.Tensor,
               action_log_probs: torch.Tensor, value_preds: torch.Tensor,
               rewards: torch.Tensor, masks: torch.Tensor, bad_masks: Optional[torch.Tensor] = None,
               active_masks: Optional[torch.Tensor] = None):
        """插入新的经验"""

        self.observations[self.step + 1].copy_(obs)
        self.states[self.step + 1].copy_(states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        if bad_masks is not None:
            self.bad_masks[self.step + 1].copy_(bad_masks)

        if active_masks is not None:
            self.active_masks[self.step + 1].copy_(active_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        """更新后的处理"""
        self.observations[0].copy_(self.observations[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.active_masks[0].copy_(self.active_masks[-1])

    def compute_returns(self, next_value: torch.Tensor, use_gae: bool = True):
        """计算回报和优势"""
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                if self.use_proper_time_limits:
                    delta = (self.rewards[step] +
                             self.gamma * self.value_preds[step + 1] * self.bad_masks[step + 1] -
                             self.value_preds[step])
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                else:
                    delta = (self.rewards[step] +
                             self.gamma * self.value_preds[step + 1] * self.masks[step + 1] -
                             self.value_preds[step])
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                self.advantages[step] = gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                if self.use_proper_time_limits:
                    self.returns[step] = (self.returns[step + 1] * self.gamma * self.bad_masks[step + 1] +
                                          self.rewards[step])
                else:
                    self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] +
                                          self.rewards[step])

            self.advantages = self.returns[:-1] - self.value_preds[:-1]

    def feed_forward_generator(self, advantages: torch.Tensor, num_mini_batch: int = None,
                               mini_batch_size: int = None):
        """生成前向传播的小批次数据"""
        num_steps, num_agents = self.rewards.size()[0:2]
        batch_size = num_agents * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                f"PPO requires the number of processes ({num_agents}) "
                f"* number of steps ({num_steps}) = {num_agents * num_steps} "
                f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch})."
            )
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True
        )

        for indices in sampler:
            obs_batch = self.observations[:-1].view(-1, *self.observations.size()[2:])[indices]
            states_batch = self.states[:-1].view(-1, *self.states.size()[2:])[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            active_masks_batch = self.active_masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, states_batch, actions_batch, value_preds_batch, return_batch, \
                masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ