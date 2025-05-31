import torch
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import logging
from tensorboardX import SummaryWriter

from model import OptimizedMAPPOPolicy
from agent import MAPPO
from memroy import RolloutBuffer
from adaptive_managers import HyperparameterManager, PhaseManager


class MAPPOTrainer:
    """MAPPO训练器"""

    def __init__(self, env, config, logger=None):
        self.env = env
        self.config = config
        self.logger = logger or self._setup_logger()

        # 环境参数
        self.num_agents = env.get_num_agents()
        self.obs_dim = env.get_obs_dim()
        self.action_dim = env.get_action_dim()
        self.state_dim = env.get_state_dim() if config.get('USE_CENTRALIZED_V') else self.obs_dim

        # 训练参数
        self.num_env_steps = config.get('TOTAL_TIMESTEPS', 10000000)
        self.episode_length = config.get('MAX_EPISODE_STEPS', 100)
        self.n_rollout_threads = config.get('NUM_ENVS', 8)
        self.n_eval_rollout_threads = 1
        self.rollout_length = config.get('ROLLOUT_LENGTH', 2048)
        self.eval_episodes = config.get('EVAL_EPISODES', 32)

        # 日志参数
        self.save_freq = config.get('SAVE_FREQ', 50000)
        self.eval_freq = config.get('EVAL_FREQ', 10000)
        self.log_freq = config.get('LOG_FREQ', 1000)

        # 模型保存路径
        self.model_dir = os.path.join(config.get('LOG_DIR', 'logs'), 'models')
        os.makedirs(self.model_dir, exist_ok=True)

        # 初始化网络
        self.policy = OptimizedMAPPOPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            hidden_dims=config.get('HIDDEN_UNITS', [256, 256]),
            use_centralized_v=config.get('USE_CENTRALIZED_V', True),
            use_attention=config.get('USE_ATTENTION', True),
            use_spatial_encoding=config.get('USE_SPATIAL_ENCODING', True),
            use_dueling=config.get('USE_DUELING', True)
        )

        # 初始化算法
        self.algorithm = MAPPO(self.policy, config)

        # 初始化缓冲区
        self.buffer = RolloutBuffer(
            num_steps=self.rollout_length,
            num_agents=self.num_agents,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            use_centralized_v=config.get('USE_CENTRALIZED_V', True),
            use_gae=True,
            gamma=config.get('GAMMA', 0.99),
            gae_lambda=config.get('GAE_LAMBDA', 0.95),
            use_proper_time_limits=config.get('USE_PROPER_TIME_LIMITS', False)
        )

        # TensorBoard日志
        self.writer = SummaryWriter(log_dir=config.get('LOG_DIR', 'logs'))

        # 训练统计
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.total_env_steps = 0

        # 添加自适应管理器
        if config.get('USE_ADAPTIVE_HYPERPARAMETERS', False):
            self.hp_manager = HyperparameterManager(config, self)
        else:
            self.hp_manager = None

        if config.get('USE_PHASED_TRAINING', False):
            self.phase_manager = PhaseManager(config)
        else:
            self.phase_manager = None

    def _setup_logger(self):
        """设置日志器"""
        logger = logging.getLogger('MAPPO')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def collect_rollouts(self):
        """收集轨迹数据"""
        self.policy.eval()

        # 重置环境和缓冲区
        obs, info = self.env.reset()
        self.buffer.reset()

        # 初始化状态
        if self.config.get('USE_CENTRALIZED_V'):
            states = info.get('centralized_obs')
            if states is None:
                states = np.concatenate([obs[agent] for agent in self.env.get_agent_ids()])
        else:
            states = np.concatenate([obs[agent] for agent in self.env.get_agent_ids()])

        # 转换为tensor
        obs_tensor = torch.from_numpy(
            np.stack([obs[agent] for agent in self.env.get_agent_ids()])
        ).float()
        states_tensor = torch.from_numpy(states).float().unsqueeze(0).repeat(self.num_agents, 1)

        # 插入初始观测
        self.buffer.observations[0].copy_(obs_tensor)
        self.buffer.states[0].copy_(states_tensor)

        episode_rewards = np.zeros(self.num_agents)
        episode_length = 0

        for step in range(self.rollout_length):
            with torch.no_grad():
                # 获取动作
                actions, action_log_probs, _ = self.policy.get_actions(
                    obs_tensor,
                    masks=self.buffer.masks[step]
                )

                # 获取价值估计
                values, _ = self.policy.get_values(
                    states_tensor,
                    masks=self.buffer.masks[step]
                )

            # 转换动作格式
            actions_dict = {}
            for i, agent in enumerate(self.env.get_agent_ids()):
                actions_dict[agent] = actions[i].numpy()

            # 执行动作
            next_obs, rewards, dones, truncated, next_info = self.env.step(actions_dict)

            # 计算奖励和mask
            reward_tensor = torch.from_numpy(
                np.array([rewards[agent] for agent in self.env.get_agent_ids()])
            ).float().unsqueeze(-1)

            done_tensor = torch.from_numpy(
                np.array([dones.get('__all__', False)] * self.num_agents)
            ).float().unsqueeze(-1)

            masks = torch.ones_like(done_tensor) - done_tensor

            # 更新统计信息
            episode_rewards += np.array([rewards[agent] for agent in self.env.get_agent_ids()])
            episode_length += 1

            # 准备下一步的观测和状态
            next_obs_tensor = torch.from_numpy(
                np.stack([next_obs[agent] for agent in self.env.get_agent_ids()])
            ).float()

            if self.config.get('USE_CENTRALIZED_V'):
                next_states = next_info.get('centralized_obs')
                if next_states is None:
                    next_states = np.concatenate([next_obs[agent] for agent in self.env.get_agent_ids()])
            else:
                next_states = np.concatenate([next_obs[agent] for agent in self.env.get_agent_ids()])

            next_states_tensor = torch.from_numpy(next_states).float().unsqueeze(0).repeat(self.num_agents, 1)

            # 插入数据到缓冲区
            self.buffer.insert(
                obs=next_obs_tensor,
                states=next_states_tensor,
                actions=actions,
                action_log_probs=action_log_probs,
                value_preds=values,
                rewards=reward_tensor,
                masks=masks
            )

            # 更新观测
            obs_tensor = next_obs_tensor
            states_tensor = next_states_tensor

            # 检查episode结束
            if dones.get('__all__', False) or truncated.get('__all__', False) or episode_length >= self.episode_length:
                self.episode_rewards.append(np.mean(episode_rewards))
                self.episode_lengths.append(episode_length)

                # 重置环境
                obs, info = self.env.reset()
                obs_tensor = torch.from_numpy(
                    np.stack([obs[agent] for agent in self.env.get_agent_ids()])
                ).float()

                if self.config.get('USE_CENTRALIZED_V'):
                    states = info.get('centralized_obs')
                    if states is None:
                        states = np.concatenate([obs[agent] for agent in self.env.get_agent_ids()])
                else:
                    states = np.concatenate([obs[agent] for agent in self.env.get_agent_ids()])

                states_tensor = torch.from_numpy(states).float().unsqueeze(0).repeat(self.num_agents, 1)

                episode_rewards = np.zeros(self.num_agents)
                episode_length = 0

        # 计算下一步的价值估计
        with torch.no_grad():
            next_values, _ = self.policy.get_values(
                states_tensor,
                masks=self.buffer.masks[-1]
            )

        # 计算回报和优势
        self.buffer.compute_returns(next_values, use_gae=True)

        self.total_env_steps += self.rollout_length

    def train_step(self):
        """执行一步训练"""
        self.policy.train()

        # 更新策略
        train_info = self.algorithm.update(
            self.buffer,
            use_obs_instead_of_state=self.config.get('USE_OBS_INSTEAD_OF_STATE', False)
        )

        # 清理缓冲区
        self.buffer.after_update()

        return train_info

    def evaluate(self):
        """评估策略"""
        self.policy.eval()

        eval_episode_rewards = []
        eval_episode_lengths = []
        eval_localization_errors = []
        eval_coordination_scores = []
        eval_convergence_rates = []

        for episode in range(self.eval_episodes):
            obs, info = self.env.reset()
            episode_reward = np.zeros(self.num_agents)
            episode_length = 0
            done = False

            while not done and episode_length < self.episode_length:
                obs_tensor = torch.from_numpy(
                    np.stack([obs[agent] for agent in self.env.get_agent_ids()])
                ).float()

                with torch.no_grad():
                    actions, _, _ = self.policy.get_actions(
                        obs_tensor,
                        deterministic=True
                    )

                actions_dict = {}
                for i, agent in enumerate(self.env.get_agent_ids()):
                    actions_dict[agent] = actions[i].numpy()

                obs, rewards, dones, truncated, info = self.env.step(actions_dict)

                episode_reward += np.array([rewards[agent] for agent in self.env.get_agent_ids()])
                episode_length += 1

                done = dones.get('__all__', False) or truncated.get('__all__', False)

            eval_episode_rewards.append(np.mean(episode_reward))
            eval_episode_lengths.append(episode_length)
            eval_localization_errors.append(info.get('mean_localization_error', 0))
            eval_coordination_scores.append(info.get('coordination_score', 0))
            eval_convergence_rates.append(info.get('convergence_rate', 0))

        return {
            'eval_mean_reward': np.mean(eval_episode_rewards),
            'eval_std_reward': np.std(eval_episode_rewards),
            'eval_mean_length': np.mean(eval_episode_lengths),
            'mean_localization_error': np.mean(eval_localization_errors),
            'coordination_score': np.mean(eval_coordination_scores),
            'convergence_rate': np.mean(eval_convergence_rates)
        }

    def save_model(self, step):
        """保存模型"""
        if isinstance(step, str):
            save_path = step
        else:
            save_path = os.path.join(self.model_dir, f'model_step_{step}.pt')
        self.algorithm.save(save_path)
        self.logger.info(f"Model saved to {save_path}")

    def log_enhanced_training_info(self, step, train_info, eval_info=None):
        """增强版训练信息记录"""
        # 原有日志记录
        for key, value in train_info.items():
            self.writer.add_scalar(f'train/{key}', value, step)

        # 记录episode统计
        if len(self.episode_rewards) > 0:
            self.writer.add_scalar('train/mean_episode_reward', np.mean(self.episode_rewards), step)
            self.writer.add_scalar('train/mean_episode_length', np.mean(self.episode_lengths), step)

        # 记录评估结果
        if eval_info:
            for key, value in eval_info.items():
                self.writer.add_scalar(f'eval/{key}', value, step)

        # 超参数信息
        if self.hp_manager:
            hp_params = self.hp_manager.get_current_params()
            for key, value in hp_params.items():
                self.writer.add_scalar(f'hyperparameters/{key}', value, step)

        # 阶段信息
        if self.phase_manager:
            progress_info = self.phase_manager.get_progress_info(step)
            if isinstance(progress_info, dict):
                self.writer.add_scalar('training/current_phase', progress_info['current_phase'], step)

                # 每1000步打印详细信息
                if step % 1000 == 0:
                    self.logger.info(f"Phase Progress: {progress_info['phase_name']} "
                                     f"({progress_info['phase_progress']})")

        # 记录学习率
        current_lr = self.algorithm.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('train/learning_rate', current_lr, step)

        # 打印日志
        if step % self.log_freq == 0:
            log_str = f"Step: {step}, Total Steps: {self.total_env_steps}"
            if len(self.episode_rewards) > 0:
                log_str += f", Mean Reward: {np.mean(self.episode_rewards):.4f}"
            if eval_info:
                log_str += f", Eval Reward: {eval_info['eval_mean_reward']:.4f}"
                log_str += f", Localization Error: {eval_info['mean_localization_error']:.4f}"
            self.logger.info(log_str)

    def train(self):
        """主训练循环 - 增强版"""
        self.logger.info("Starting enhanced MAPPO training...")

        step = 0
        start_time = time.time()

        while self.total_env_steps < self.num_env_steps:
            # 收集轨迹
            self.collect_rollouts()

            # 训练策略
            train_info = self.train_step()

            step += 1

            # 评估
            eval_info = None
            if step % (self.eval_freq // self.rollout_length) == 0:
                eval_info = self.evaluate()

            # 更新超参数
            if self.hp_manager:
                self.hp_manager.update(step, train_info, eval_info)

            # 检查阶段转换
            if self.phase_manager and eval_info:
                if self.phase_manager.check_phase_transition(step, eval_info):
                    phase_advanced = self.phase_manager.advance_phase(step, self)
                    if not phase_advanced:
                        self.logger.info("All training phases completed!")
                        break

            # 记录增强版日志
            self.log_enhanced_training_info(step, train_info, eval_info)

            # 保存模型
            if step % (self.save_freq // self.rollout_length) == 0:
                self.save_model(step)

            # 早停检查
            if self.hp_manager and self.hp_manager.should_early_stop():
                self.logger.info(f"Early stopping triggered at step {step}")
                break

        # 保存最终模型
        self.save_model(step)

        total_time = time.time() - start_time
        self.logger.info(f"Enhanced training completed in {total_time:.2f} seconds")

        self.writer.close()