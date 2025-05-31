import gym
from gym import spaces
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from collections import defaultdict


class SensorNetworkLocalizationEnv(gym.Env):
    """
    多智能体传感器网络定位环境 - MAPPO版本
    基于连续动作空间和分层奖励结构的强化学习环境
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 anchors_pos: np.ndarray,
                 sensor_pos: np.ndarray,
                 estimated_positions: np.ndarray,
                 communication_range: float = 3.0,
                 noise_std: float = 0.1,
                 max_episode_steps: int = 100,
                 initial_pos_bounds: np.ndarray = np.array([[-50.0, 300.0], [-50.0, 250.0]]),
                 render_mode: str = None,
                 dimension: int = 2,
                 reward_scale: float = 1.0,
                 global_reward_freq: int = 5,
                 use_centralized_critic: bool = True):

        super(SensorNetworkLocalizationEnv, self).__init__()
        self.anchors_pos = np.array(anchors_pos, dtype=np.float32)
        self.sensor_pos = np.array(sensor_pos, dtype=np.float32)
        self.n_sensors = self.sensor_pos.shape[0]
        self.n_anchors = self.anchors_pos.shape[0]

        self.initial_pos_bounds = initial_pos_bounds
        self.communication_range = communication_range
        self.noise_std = noise_std
        self.max_episode_steps = max_episode_steps
        self.dimension = dimension
        self.reward_scale = reward_scale
        self.global_reward_freq = global_reward_freq
        self.use_centralized_critic = use_centralized_critic

        # MAPPO需要的智能体列表
        self.agents = [f'sensor_{i}' for i in range(self.n_sensors)]
        self.possible_agents = self.agents.copy()

        # 存储动作历史
        self.last_actions = np.zeros((self.n_sensors, dimension), dtype=np.float32)

        # 动作空间：连续位移向量
        self.action_space = spaces.Dict({
            f'sensor_{i}': spaces.Box(
                low=-1.0, high=1.0,
                shape=(dimension,), dtype=np.float32
            ) for i in range(self.n_sensors)
        })

        # 单一动作空间接口
        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(dimension,), dtype=np.float32
        )

        # 状态空间维度计算
        max_neighbors = self.n_sensors - 1
        max_anchors_per_sensor = self.n_anchors

        state_dim_per_sensor = (
                dimension +  # 当前位置估计
                max_neighbors * (dimension + 1) +  # 邻居相对位置 + 距离测量
                max_anchors_per_sensor * (dimension + 1) +  # 锚点相对位置 + 距离测量
                dimension  # 最近一步位移
        )

        self.observation_space = spaces.Dict({
            f'sensor_{i}': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(state_dim_per_sensor,), dtype=np.float32
            ) for i in range(self.n_sensors)
        })

        # 单一观测空间接口
        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim_per_sensor,), dtype=np.float32
        )

        # 中心化观测空间
        if self.use_centralized_critic:
            centralized_obs_dim = (
                    self.n_sensors * state_dim_per_sensor +  # 所有智能体的局部观测
                    self.n_anchors * dimension +  # 锚点位置
                    self.n_sensors * dimension  # 真实位置（训练时可用）
            )
            self.centralized_observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(centralized_obs_dim,), dtype=np.float32
            )

        # 环境状态
        self.render_mode = render_mode
        self.true_positions = None
        self.estimated_positions = estimated_positions
        self.anchor_positions = None
        self.communication_graph = None
        self.distance_measurements = None
        self.anchor_measurements = None

        self.current_step = 0
        self.max_neighbors = max_neighbors
        self.max_anchors_per_sensor = max_anchors_per_sensor

        # 奖励权重（支持动态调整）
        self.reward_weights = {
            'local_alignment': 1.0,
            'motion_penalty': 1.0,
            'global_consistency': 1.0,
            'boundary_penalty': 1.0
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """重置环境"""
        super().reset(seed=seed)
        self.current_step = 0
        self.last_actions = np.zeros((self.n_sensors, self.dimension), dtype=np.float32)

        # 生成真实位置
        self.true_positions = self.sensor_pos.copy()

        # 生成锚点位置
        self.anchor_positions = self.anchors_pos.copy()

        # 初始化估计位置
        initial_noise = np.random.normal(0, self.noise_std * 2, size=self.true_positions.shape)
        self.estimated_positions = self.true_positions + initial_noise

        # 构建通信图
        self.communication_graph = self._build_communication_graph()

        # 生成距离测量
        self._generate_measurements()

        observations = self._get_observations()

        # MAPPO兼容性信息
        info = {
            'centralized_obs': self.get_centralized_observation() if self.use_centralized_critic else None,
            'agent_mask': [True] * self.n_sensors,
            'mean_localization_error': np.mean(np.linalg.norm(self.estimated_positions - self.true_positions, axis=1))
        }

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """执行一步动作"""
        self.current_step += 1

        # 执行动作
        for i in range(self.n_sensors):
            action = actions[f'sensor_{i}']
            scaled_action = action * self.communication_range * 0.1
            self.last_actions[i] = scaled_action

            # 更新位置并检查边界
            new_pos = self.estimated_positions[i] + scaled_action
            new_pos = self._clip_to_bounds(new_pos)
            self.estimated_positions[i] = new_pos

        # 重新生成距离测量
        self._generate_measurements()

        # 计算奖励
        rewards = self._compute_rewards()

        # 检查终止条件
        done = self._check_done()
        truncated = {'__all__': self.current_step >= self.max_episode_steps}

        # 获取新观测
        observations = self._get_observations()

        # 计算信息
        info = self._get_info()

        # 添加MAPPO需要的信息
        localization_errors = np.linalg.norm(self.estimated_positions - self.true_positions, axis=1)
        info.update({
            'centralized_obs': self.get_centralized_observation() if self.use_centralized_critic else None,
            'agent_mask': [True] * self.n_sensors,
            'individual_rewards': list(rewards.values()),
            'global_reward': sum(rewards.values()) / len(rewards),
            'mean_localization_error': np.mean(localization_errors),
            'max_localization_error': np.max(localization_errors),
            'boundary_violation_rate': self._get_boundary_violation_rate(),
            'coordination_score': self._get_coordination_score(),
            'convergence_rate': self._get_convergence_rate()
        })

        return observations, rewards, done, truncated, info

    def _clip_to_bounds(self, position):
        """将位置限制在边界内"""
        clipped = position.copy()
        clipped[0] = np.clip(clipped[0], self.initial_pos_bounds[0, 0], self.initial_pos_bounds[0, 1])
        clipped[1] = np.clip(clipped[1], self.initial_pos_bounds[1, 0], self.initial_pos_bounds[1, 1])
        return clipped

    def _get_boundary_violation_rate(self):
        """计算边界违反率"""
        violations = 0
        for pos in self.estimated_positions:
            if (pos[0] <= self.initial_pos_bounds[0, 0] or pos[0] >= self.initial_pos_bounds[0, 1] or
                    pos[1] <= self.initial_pos_bounds[1, 0] or pos[1] >= self.initial_pos_bounds[1, 1]):
                violations += 1
        return violations / self.n_sensors

    def _get_coordination_score(self):
        """计算协调得分"""
        if not self.communication_graph.edges():
            return 0.0

        total_error = 0.0
        count = 0

        for edge in self.communication_graph.edges():
            i, j = edge
            if (i, j) in self.distance_measurements:
                measured_dist = self.distance_measurements[(i, j)]
                estimated_dist = np.linalg.norm(self.estimated_positions[i] - self.estimated_positions[j])
                error = abs(measured_dist - estimated_dist)
                total_error += error
                count += 1

        if count == 0:
            return 0.0

        avg_error = total_error / count
        return max(0.0, 1.0 - avg_error / self.communication_range)

    def _get_convergence_rate(self):
        """计算收敛率"""
        errors = np.linalg.norm(self.estimated_positions - self.true_positions, axis=1)
        converged = np.sum(errors < self.communication_range * 0.1)
        return converged / self.n_sensors

    def get_centralized_observation(self) -> np.ndarray:
        """获取中心化观测"""
        if not self.use_centralized_critic:
            return None

        # 获取所有智能体的局部观测
        local_observations = self._get_observations()
        all_local_obs = []
        for agent in self.agents:
            all_local_obs.append(local_observations[agent])

        # 构建中心化观测
        centralized_obs = np.concatenate([
            np.concatenate(all_local_obs),  # 所有局部观测
            self.anchor_positions.flatten(),  # 锚点位置
            self.true_positions.flatten(),  # 真实位置（训练时可用）
        ])

        return centralized_obs.astype(np.float32)

    def get_agent_ids(self):
        """返回所有智能体ID"""
        return self.agents

    def get_num_agents(self):
        """返回智能体数量"""
        return self.n_sensors

    def get_action_dim(self):
        """返回单个智能体的动作维度"""
        return self.dimension

    def get_obs_dim(self):
        """返回单个智能体的观测维度"""
        return self.single_observation_space.shape[0]

    def get_state_dim(self):
        """返回全局状态维度"""
        if self.use_centralized_critic:
            return self.centralized_observation_space.shape[0]
        else:
            return self.get_obs_dim() * self.n_sensors

    def update_reward_weights(self, new_weights):
        """更新奖励权重"""
        self.reward_weights.update(new_weights)

    def _build_communication_graph(self) -> nx.Graph:
        """构建通信图"""
        G = nx.Graph()
        G.add_nodes_from(range(self.n_sensors))

        for i in range(self.n_sensors):
            for j in range(i + 1, self.n_sensors):
                distance = np.linalg.norm(
                    self.true_positions[i] - self.true_positions[j]
                )
                if distance <= self.communication_range:
                    G.add_edge(i, j)

        return G

    def _generate_measurements(self):
        """生成距离测量"""
        # 传感器间距离测量
        self.distance_measurements = {}
        for edge in self.communication_graph.edges():
            i, j = edge
            true_distance = np.linalg.norm(
                self.true_positions[i] - self.true_positions[j]
            )
            noise = np.random.normal(0, self.noise_std)
            self.distance_measurements[(i, j)] = true_distance + noise

        # 锚点距离测量
        self.anchor_measurements = {}
        for i in range(self.n_sensors):
            for k in range(self.n_anchors):
                distance = np.linalg.norm(
                    self.true_positions[i] - self.anchor_positions[k]
                )
                if distance <= self.communication_range * 1.5:
                    noise = np.random.normal(0, self.noise_std)
                    self.anchor_measurements[(i, k)] = distance + noise

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """获取观测"""
        observations = {}

        for i in range(self.n_sensors):
            # 当前位置估计
            current_pos = self.estimated_positions[i]

            # 获取邻居信息
            neighbors = list(self.communication_graph.neighbors(i))
            neighbor_info = np.zeros((self.max_neighbors, self.dimension + 1))

            for idx, neighbor in enumerate(neighbors[:self.max_neighbors]):
                rel_pos = self.estimated_positions[neighbor] - current_pos
                neighbor_info[idx, :self.dimension] = rel_pos

                if (i, neighbor) in self.distance_measurements:
                    neighbor_info[idx, self.dimension] = self.distance_measurements[(i, neighbor)]
                elif (neighbor, i) in self.distance_measurements:
                    neighbor_info[idx, self.dimension] = self.distance_measurements[(neighbor, i)]

            # 获取锚点信息
            anchor_info = np.zeros((self.max_anchors_per_sensor, self.dimension + 1))
            anchor_idx = 0
            for k in range(self.n_anchors):
                if (i, k) in self.anchor_measurements and anchor_idx < self.max_anchors_per_sensor:
                    rel_pos = self.anchor_positions[k] - current_pos
                    anchor_info[anchor_idx, :self.dimension] = rel_pos
                    anchor_info[anchor_idx, self.dimension] = self.anchor_measurements[(i, k)]
                    anchor_idx += 1

            # 运动历史
            motion_history = self.last_actions[i]

            # 组合状态
            state = np.concatenate([
                current_pos,
                neighbor_info.flatten(),
                anchor_info.flatten(),
                motion_history
            ])

            observations[f'sensor_{i}'] = state.astype(np.float32)

        return observations

    def _compute_rewards(self) -> Dict[str, float]:
        """计算分层奖励结构"""
        rewards = {}

        # 计算各类奖励
        local_rewards = self._compute_local_rewards()
        motion_penalties = self._compute_motion_penalties()
        boundary_penalties = self._compute_boundary_penalties()

        if self.current_step % self.global_reward_freq == 0:
            global_reward = self._compute_global_consistency_reward()
        else:
            global_reward = 0.0

        # 组合奖励
        for i, agent in enumerate(self.agents):
            total_reward = (
                    self.reward_weights['local_alignment'] * local_rewards[i] +
                    self.reward_weights['motion_penalty'] * motion_penalties[i] +
                    self.reward_weights['boundary_penalty'] * boundary_penalties[i] +
                    self.reward_weights['global_consistency'] * global_reward
            )
            rewards[agent] = total_reward * self.reward_scale

        return rewards

    def _compute_local_rewards(self) -> List[float]:
        """计算局部对齐奖励"""
        rewards = [0.0] * self.n_sensors
        baseline = 0.1 * self.communication_range

        for i in range(self.n_sensors):
            # 与邻居的测量误差
            neighbors = list(self.communication_graph.neighbors(i))
            for neighbor in neighbors:
                if (i, neighbor) in self.distance_measurements:
                    measured_distance = self.distance_measurements[(i, neighbor)]
                elif (neighbor, i) in self.distance_measurements:
                    measured_distance = self.distance_measurements[(neighbor, i)]
                else:
                    continue

                estimated_distance = np.linalg.norm(
                    self.estimated_positions[i] - self.estimated_positions[neighbor]
                )

                error = abs(measured_distance - estimated_distance) - baseline
                weight = 1.0 / (self.noise_std ** 2)
                rewards[i] -= weight * error

            # 与锚点的测量误差
            for k in range(self.n_anchors):
                if (i, k) in self.anchor_measurements:
                    measured_distance = self.anchor_measurements[(i, k)]
                    estimated_distance = np.linalg.norm(
                        self.estimated_positions[i] - self.anchor_positions[k]
                    )

                    error = abs(measured_distance - estimated_distance) - baseline
                    weight = 1.0 / (self.noise_std ** 2)
                    rewards[i] -= weight * error

        return rewards

    def _compute_motion_penalties(self) -> List[float]:
        """计算运动惩罚"""
        penalties = []
        motion_penalty_coeff = 0.01

        for i in range(self.n_sensors):
            motion_magnitude = np.linalg.norm(self.last_actions[i])
            penalties.append(-motion_penalty_coeff * motion_magnitude)

        return penalties

    def _compute_boundary_penalties(self) -> List[float]:
        """计算边界惩罚"""
        penalties = []
        boundary_penalty_coeff = 1.0

        for i in range(self.n_sensors):
            pos = self.estimated_positions[i]
            penalty = 0.0

            # 检查边界违反
            if pos[0] <= self.initial_pos_bounds[0, 0] or pos[0] >= self.initial_pos_bounds[0, 1]:
                penalty += boundary_penalty_coeff
            if pos[1] <= self.initial_pos_bounds[1, 0] or pos[1] >= self.initial_pos_bounds[1, 1]:
                penalty += boundary_penalty_coeff

            penalties.append(-penalty)

        return penalties

    def _compute_global_consistency_reward(self) -> float:
        """计算全局一致性奖励"""
        if not self.communication_graph.edges():
            return 0.0

        total_consistency = 0.0
        count = 0

        for edge in self.communication_graph.edges():
            i, j = edge
            if (i, j) in self.distance_measurements:
                true_dist = self.distance_measurements[(i, j)]
            elif (j, i) in self.distance_measurements:
                true_dist = self.distance_measurements[(j, i)]
            else:
                continue

            est_dist = np.linalg.norm(self.estimated_positions[i] - self.estimated_positions[j])
            consistency = np.exp(-(true_dist - est_dist) ** 2 / (2 * self.noise_std ** 2))
            total_consistency += consistency
            count += 1

        if count == 0:
            return 0.0

        avg_consistency = total_consistency / count
        return avg_consistency

    def _check_done(self) -> Dict[str, bool]:
        """检查终止条件"""
        done = {'__all__': False}

        # 收敛检查
        localization_errors = np.linalg.norm(
            self.estimated_positions - self.true_positions, axis=1
        )
        max_error = np.max(localization_errors)

        if max_error < self.communication_range * 0.05:
            done['__all__'] = True

        return done

    def _get_info(self) -> Dict:
        """获取额外信息"""
        localization_errors = np.linalg.norm(
            self.estimated_positions - self.true_positions, axis=1
        )

        return {
            'step': self.current_step,
            'true_positions': self.true_positions.copy(),
            'estimated_positions': self.estimated_positions.copy()
        }

    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            plt.figure(figsize=(12, 10))

            # 绘制真实位置
            plt.scatter(self.true_positions[:, 0], self.true_positions[:, 1],
                        c='blue', marker='o', s=100, label='True Positions', alpha=0.7)

            # 绘制估计位置
            plt.scatter(self.estimated_positions[:, 0], self.estimated_positions[:, 1],
                        c='red', marker='x', s=100, label='Estimated Positions')

            # 绘制锚点
            plt.scatter(self.anchor_positions[:, 0], self.anchor_positions[:, 1],
                        c='green', marker='s', s=150, label='Anchors')

            # 绘制通信链接
            for edge in self.communication_graph.edges():
                i, j = edge
                plt.plot([self.true_positions[i, 0], self.true_positions[j, 0]],
                         [self.true_positions[i, 1], self.true_positions[j, 1]],
                         'gray', alpha=0.3, linewidth=1)

            # 绘制误差线
            for i in range(self.n_sensors):
                plt.plot([self.true_positions[i, 0], self.estimated_positions[i, 0]],
                         [self.true_positions[i, 1], self.estimated_positions[i, 1]],
                         'orange', alpha=0.5, linewidth=2)

            x_min, x_max = self.initial_pos_bounds[0]
            y_min, y_max = self.initial_pos_bounds[1]

            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1

            plt.xlim(x_min - x_margin, x_max + x_margin)
            plt.ylim(y_min - y_margin, y_max + y_margin)

            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            plt.title(f'Sensor Network Localization - Step {self.current_step}\n'
                      f'Mean Error: {np.mean(np.linalg.norm(self.estimated_positions - self.true_positions, axis=1)):.2f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

    def close(self):
        """关闭环境"""
        pass


class MAPPOWrapper:
    """MAPPO环境包装器"""

    def __init__(self, env):
        self.env = env
        self.agents = [f'sensor_{i}' for i in range(env.n_sensors)]
        self.num_agents = env.n_sensors

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(seed=seed)
        return obs, info

    def step(self, actions: Dict[str, np.ndarray]):
        return self.env.step(actions)

    def render(self, mode: str = 'human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    @property
    def observation_space(self):
        return self.env.single_observation_space

    @property
    def action_space(self):
        return self.env.single_action_space

    def get_centralized_observation(self):
        return self.env.get_centralized_observation()

    def get_agent_ids(self):
        return self.env.get_agent_ids()

    def get_num_agents(self):
        return self.env.get_num_agents()

    def get_obs_dim(self):
        return self.env.get_obs_dim()

    def get_action_dim(self):
        return self.env.get_action_dim()

    def get_state_dim(self):
        return self.env.get_state_dim()
