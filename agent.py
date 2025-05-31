import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional


class MAPPO:
    """Multi-Agent Proximal Policy Optimization算法"""

    def __init__(self, policy, config):
        self.policy = policy
        self.config = config

        # 算法参数
        self.eps_clip = config.get('EPS_CLIP', 0.2)
        self.ppo_epochs = config.get('PPO_EPOCHS', 15)
        self.num_mini_batch = config.get('NUM_MINI_BATCH', 1)
        self.value_loss_coef = config.get('VALUE_LOSS_COEF', 0.5)
        self.entropy_coef = config.get('ENTROPY_COEF', 0.01)
        self.max_grad_norm = config.get('MAX_GRAD_NORM', 0.5)
        self.use_clipped_value_loss = config.get('USE_CLIPPED_VALUE_LOSS', True)
        self.use_huber_loss = config.get('USE_HUBER_LOSS', True)
        self.huber_delta = config.get('HUBER_DELTA', 10.0)
        self.use_value_active_masks = config.get('USE_VALUE_ACTIVE_MASKS', True)
        self.use_policy_active_masks = config.get('USE_POLICY_ACTIVE_MASKS', True)

        # 优化器
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.get('ACTOR_LR', 5e-4),
            eps=1e-5
        )

        # 学习率调度器
        if config.get('USE_LINEAR_LR_DECAY', True):
            total_updates = config.get('TOTAL_TIMESTEPS', 10000000) // config.get('ROLLOUT_LENGTH', 2048)
            self.lr_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=total_updates
            )
        else:
            self.lr_scheduler = None

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

    def update(self, rollout_buffer, use_obs_instead_of_state=False):
        """更新策略网络"""
        # 计算优势
        advantages = rollout_buffer.advantages
        if self.config.get('USE_FEATURE_NORMALIZATION', True):
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        value_loss_epoch = 0
        action_loss_epoch = 0
        entropy_loss_epoch = 0
        kl_div_epoch = 0

        for epoch in range(self.ppo_epochs):
            data_generator = rollout_buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                (obs_batch, states_batch, actions_batch, value_preds_batch, return_batch,
                 masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ) = sample

                # 移动到设备
                obs_batch = obs_batch.to(self.device)
                states_batch = states_batch.to(self.device)
                actions_batch = actions_batch.to(self.device)
                value_preds_batch = value_preds_batch.to(self.device)
                return_batch = return_batch.to(self.device)
                masks_batch = masks_batch.to(self.device)
                active_masks_batch = active_masks_batch.to(self.device)
                old_action_log_probs_batch = old_action_log_probs_batch.to(self.device)
                adv_targ = adv_targ.to(self.device)

                # 评估动作和状态
                eval_input = states_batch if not use_obs_instead_of_state else obs_batch
                values, action_log_probs, entropy = self.policy.evaluate_actions(
                    obs_batch, eval_input, actions_batch
                )

                # 计算策略损失
                imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

                surr1 = imp_weights * adv_targ
                surr2 = torch.clamp(imp_weights, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_targ

                if self.use_policy_active_masks:
                    policy_action_loss = (-torch.min(surr1,
                                                     surr2) * active_masks_batch).sum() / active_masks_batch.sum()
                else:
                    policy_action_loss = -torch.min(surr1, surr2).mean()

                # 计算价值损失
                value_pred_clipped = value_preds_batch + \
                                     (values - value_preds_batch).clamp(-self.eps_clip, self.eps_clip)

                if self.use_huber_loss:
                    value_losses = self.huber_loss(values, return_batch, self.huber_delta)
                    value_losses_clipped = self.huber_loss(value_pred_clipped, return_batch, self.huber_delta)
                else:
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)

                if self.use_clipped_value_loss:
                    value_loss = torch.max(value_losses, value_losses_clipped)
                else:
                    value_loss = value_losses

                if self.use_value_active_masks:
                    value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
                else:
                    value_loss = value_loss.mean()

                # 计算熵损失
                if self.use_policy_active_masks:
                    entropy_loss = (entropy * active_masks_batch).sum() / active_masks_batch.sum()
                else:
                    entropy_loss = entropy.mean()

                # 计算KL散度
                kl_div = (old_action_log_probs_batch - action_log_probs).mean()

                # 总损失
                total_loss = (policy_action_loss +
                              self.value_loss_coef * value_loss -
                              self.entropy_coef * entropy_loss)

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += policy_action_loss.item()
                entropy_loss_epoch += entropy_loss.item()
                kl_div_epoch += kl_div.item()

        # 更新学习率
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        num_updates = self.ppo_epochs * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        entropy_loss_epoch /= num_updates
        kl_div_epoch /= num_updates

        return {
            'value_loss': value_loss_epoch,
            'action_loss': action_loss_epoch,
            'entropy_loss': entropy_loss_epoch,
            'kl_divergence': kl_div_epoch,
            'total_loss': value_loss_epoch + action_loss_epoch - entropy_loss_epoch
        }

    def huber_loss(self, pred, target, delta=1.0):
        """Huber损失函数"""
        residual = torch.abs(pred - target)
        condition = residual < delta
        small_res = 0.5 * residual.pow(2)
        large_res = delta * residual - 0.5 * delta * delta
        return torch.where(condition, small_res, large_res)

    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.lr_scheduler and checkpoint['lr_scheduler_state_dict']:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])