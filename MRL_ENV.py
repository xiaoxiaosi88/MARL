import gym
from gym import spaces
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from collections import defaultdict


class SensorNetworkLocalizationEnv(gym.Env):
    """
    多智能体传感器网络定位环境
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
                 global_reward_freq: int = 5):  # 新增参数：全局奖励计算频率

        """
        初始化传感器网络定位环境

        Args:
            anchors_pos: 锚点位置
            sensor_pos: 传感器真实位置
            estimated_positions: 初始估计位置
            communication_range: 通信范围
            noise_std: 噪声标准差
            max_episode_steps: 最大步数
            initial_pos_bounds: 位置边界范围 [[x_min, x_max], [y_min, y_max]]
            render_mode: 渲染模式
            dimension: 空间维度（2D或3D）
            reward_scale: 奖励缩放因子
            global_reward_freq: 全局奖励计算频率（每多少步计算一次）
        """
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
        self.global_reward_freq = global_reward_freq  # 全局奖励计算频率

        # MADDPG需要的智能体列表
        self.agents = [f'sensor_{i}' for i in range(self.n_sensors)]

        # 存储动作历史（用于运动惩罚）
        self.last_actions = np.zeros((self.n_sensors, dimension), dtype=np.float32)

        # 动作空间：连续位移向量 Δx_i ∈ ℝ^D
        self.action_space = spaces.Dict({
            f'sensor_{i}': spaces.Box(
                low=-1.0, high=1.0,  # 归一化动作范围
                shape=(dimension,), dtype=np.float32
            ) for i in range(self.n_sensors)
        })

        # 状态空间维度计算 - 根据设计修改
        max_neighbors = self.n_sensors - 1  # 最大邻居数
        max_anchors_per_sensor = self.n_anchors  # 最大锚点数

        # 根据设计的状态空间修改
        state_dim_per_sensor = (
                dimension +  # 当前位置估计
                max_neighbors * (dimension + 1) +  # 邻居相对位置 + 距离测量 + 噪声标准差
                max_anchors_per_sensor * (dimension + 1) +  # 锚点相对位置 + 距离测量 + 噪声标准差
                dimension  # 最近一步位移 (motion_history)
        )

        self.observation_space = spaces.Dict({
            f'sensor_{i}': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(state_dim_per_sensor,), dtype=np.float32
            ) for i in range(self.n_sensors)
        })

        # 环境状态
        self.render_mode = render_mode
        self.true_positions = None  # 真实位置
        self.estimated_positions = estimated_positions  # 估计位置
        self.anchor_positions = None  # 锚点位置
        self.communication_graph = None  # 通信图
        self.distance_measurements = None  # 距离测量
        self.anchor_measurements = None  # 锚点距离测量

        self.current_step = 0
        self.max_neighbors = max_neighbors
        self.max_anchors_per_sensor = max_anchors_per_sensor

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """重置环境"""
        super().reset(seed=seed)
        self.current_step = 0
        self.last_actions = np.zeros((self.n_sensors, self.dimension), dtype=np.float32)

        # 生成真实位置
        self.true_positions = self.sensor_pos

        # 生成锚点位置
        self.anchor_positions = self.anchors_pos

        # 初始化估计位置
        initial_noise = np.random.normal(0, self.noise_std * 2, size=self.true_positions.shape)
        self.estimated_positions = self.true_positions + initial_noise

        # 构建通信图
        self.communication_graph = self._build_communication_graph()

        # 生成距离测量
        self._generate_measurements()

        return self._get_observations()

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        """执行一步动作"""
        self.current_step += 1

        # 存储动作（用于奖励计算中的运动惩罚）
        for i in range(self.n_sensors):
            action = actions[f'sensor_{i}']
            # 缩放动作（根据环境范围）
            scaled_action = action * self.communication_range * 0.1
            self.last_actions[i] = scaled_action
            self.estimated_positions[i] += scaled_action

        # 重新生成距离测量
        self._generate_measurements()

        # 计算奖励
        rewards = self._compute_rewards()  # 修改为新的奖励计算函数

        # 检查终止条件
        done = self._check_done()

        # 获取新观测
        observations = self._get_observations()

        # 计算信息
        info = self._get_info()

        return observations, rewards, done, info

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
                if distance <= self.communication_range * 1.5:  # 锚点通信范围更大
                    noise = np.random.normal(0, self.noise_std)
                    self.anchor_measurements[(i, k)] = distance + noise

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """获取观测 - 根据设计修改"""
        observations = {}

        for i in range(self.n_sensors):
            # 当前位置估计
            current_pos = self.estimated_positions[i]

            # 获取邻居信息 (使用相对位置)
            neighbors = list(self.communication_graph.neighbors(i))
            neighbor_info = np.zeros((self.max_neighbors, self.dimension + 1))  # 相对位置 + 距离测量
            neighbor_mask = np.zeros(self.max_neighbors)

            for idx, neighbor in enumerate(neighbors[:self.max_neighbors]):
                rel_pos = self.estimated_positions[neighbor] - current_pos
                neighbor_info[idx, :self.dimension] = rel_pos

                # 使用测量距离
                if (i, neighbor) in self.distance_measurements:
                    neighbor_info[idx, self.dimension] = self.distance_measurements[(i, neighbor)]
                elif (neighbor, i) in self.distance_measurements:
                    neighbor_info[idx, self.dimension] = self.distance_measurements[(neighbor, i)]

                neighbor_mask[idx] = 1.0

            # 获取锚点信息 (使用相对位置)
            anchor_info = np.zeros((self.max_anchors_per_sensor, self.dimension + 1))  # 相对位置 + 距离测量
            anchor_mask = np.zeros(self.max_anchors_per_sensor)

            anchor_idx = 0
            for k in range(self.n_anchors):
                if (i, k) in self.anchor_measurements and anchor_idx < self.max_anchors_per_sensor:
                    rel_pos = self.anchor_positions[k] - current_pos
                    anchor_info[anchor_idx, :self.dimension] = rel_pos
                    anchor_info[anchor_idx, self.dimension] = self.anchor_measurements[(i, k)]
                    anchor_mask[anchor_idx] = 1.0
                    anchor_idx += 1

            # 运动历史 (最近一步位移)
            motion_history = self.last_actions[i]

            # 组合状态 (根据设计)
            state = np.concatenate([
                current_pos,  # 自身位置
                neighbor_info.flatten(),  # 邻居信息 (相对位置 + 距离)
                anchor_info.flatten(),  # 锚点信息 (相对位置 + 距离)
                motion_history  # 最近一步位移
            ])

            observations[f'sensor_{i}'] = state.astype(np.float32)

        return observations

    def get_global_state(self):
        """
        获取全局状态 - MADDPG的Critic需要
        返回所有智能体的观测和动作的组合
        """
        global_obs = []
        for agent in self.agents:
            agent_idx = int(agent.split('_')[1])
            global_obs.append(self.estimated_positions[agent_idx])

        # 构建全局状态，包含：
        # 1. 所有智能体的当前估计位置
        # 2. 所有锚点位置
        # 3. 所有智能体的真实位置（用于中心化训练）
        global_state = np.concatenate([
            np.array(global_obs).flatten(),  # 所有智能体估计位置
            self.anchor_positions.flatten(),  # 锚点位置
            self.true_positions.flatten(),  # 真实位置（训练时可用）
        ])

        return global_state

    def get_joint_action_space_size(self):
        """返回联合动作空间大小"""
        single_action_size = self.action_space[self.agents[0]].shape[0]
        return single_action_size * len(self.agents)

    def get_global_action_from_dict(self, actions_dict):
        """
        将字典形式的动作转换为全局动作向量

        Args:
            actions_dict: {agent_name: action_vector} 形式的动作字典

        Returns:
            np.ndarray: 全局动作向量
        """
        global_action = []
        for agent in self.agents:
            global_action.append(actions_dict[agent])
        return np.concatenate(global_action)

    def get_global_state_dim(self):
        """返回全局状态维度"""
        # 所有智能体估计位置 + 锚点位置 + 真实位置
        return (self.n_sensors * self.dimension +  # 估计位置
                self.n_anchors * self.dimension +  # 锚点位置
                self.n_sensors * self.dimension)  # 真实位置

    def get_local_observation_dim(self):
        """返回单个智能体的观测维度"""
        agent_name = self.agents[0]
        return self.observation_space[agent_name].shape[0]

    def _compute_rewards(self) -> Dict[str, float]:
        """
        计算分层奖励结构：
        1. 局部对齐奖励 (邻居和锚点)
        2. 运动惩罚
        3. 全局一致性奖励 (周期性计算)
        """
        rewards = {}

        # 1. 计算局部奖励
        local_rewards = self._compute_local_rewards()

        # 2. 运动惩罚
        motion_penalties = self._compute_motion_penalties()

        # 3. 全局一致性奖励 (周期性计算)
        if self.current_step % self.global_reward_freq == 0:
            global_reward = self._compute_global_consistency_reward()
        else:
            global_reward = 0.0

        # 组合奖励
        for i, agent in enumerate(self.agents):
            total_reward = (
                    local_rewards[i] +
                    motion_penalties[i] +
                    global_reward
            )
            rewards[agent] = total_reward * self.reward_scale

        return rewards

    def _compute_local_rewards(self) -> List[float]:
        """计算局部对齐奖励"""
        rewards = [0.0] * self.n_sensors
        baseline = 0.1 * self.communication_range  # 基线值

        for i in range(self.n_sensors):
            # 与邻居的测量误差
            neighbors = list(self.communication_graph.neighbors(i))
            for neighbor in neighbors:
                # 获取测量距离
                if (i, neighbor) in self.distance_measurements:
                    measured_distance = self.distance_measurements[(i, neighbor)]
                elif (neighbor, i) in self.distance_measurements:
                    measured_distance = self.distance_measurements[(neighbor, i)]
                else:
                    continue

                # 计算估计距离
                estimated_distance = np.linalg.norm(
                    self.estimated_positions[i] - self.estimated_positions[neighbor]
                )

                # 计算加权误差
                error = abs(measured_distance - estimated_distance) - baseline
                weight = 1.0 / (self.noise_std ** 2)  # 权重与噪声方差成反比
                rewards[i] -= weight * error

            # 与锚点的测量误差
            for k in range(self.n_anchors):
                if (i, k) in self.anchor_measurements:
                    measured_distance = self.anchor_measurements[(i, k)]
                    estimated_distance = np.linalg.norm(
                        self.estimated_positions[i] - self.anchor_positions[k]
                    )

                    # 计算加权误差
                    error = abs(measured_distance - estimated_distance) - baseline
                    weight = 1.0 / (self.noise_std ** 2)  # 权重与噪声方差成反比
                    rewards[i] -= weight * error

        return rewards

    def _compute_motion_penalties(self) -> List[float]:
        """计算运动惩罚"""
        penalties = []
        motion_penalty_coeff = 0.01  # 运动惩罚系数

        for i in range(self.n_sensors):
            # 惩罚位移大小
            motion_magnitude = np.linalg.norm(self.last_actions[i])
            penalties.append(-motion_penalty_coeff * motion_magnitude)

        return penalties

    def _compute_global_consistency_reward(self) -> float:
        """计算全局一致性奖励"""
        if not self.communication_graph.edges():
            return 0.0

        total_consistency = 0.0
        count = 0

        # 构建当前全局距离矩阵
        est_dists = np.zeros((self.n_sensors, self.n_sensors))
        for i in range(self.n_sensors):
            for j in range(i + 1, self.n_sensors):
                dist = np.linalg.norm(self.estimated_positions[i] - self.estimated_positions[j])
                est_dists[i, j] = dist
                est_dists[j, i] = dist

        # 计算拓扑一致性
        for edge in self.communication_graph.edges():
            i, j = edge
            if (i, j) in self.distance_measurements:
                true_dist = self.distance_measurements[(i, j)]
            elif (j, i) in self.distance_measurements:
                true_dist = self.distance_measurements[(j, i)]
            else:
                continue

            est_dist = est_dists[i, j]
            # 使用高斯核函数计算一致性
            consistency = np.exp(-(true_dist - est_dist) ** 2 / (2 * self.noise_std ** 2))
            total_consistency += consistency
            count += 1

        if count == 0:
            return 0.0

        avg_consistency = total_consistency / count
        return avg_consistency

    def _check_done(self) -> Dict[str, bool]:
        """检查终止条件"""
        done = {'__all__': self.current_step >= self.max_episode_steps}

        # 收敛检查
        localization_errors = np.linalg.norm(
            self.estimated_positions - self.true_positions, axis=1
        )
        max_error = np.max(localization_errors)

        # 如果最大定位误差小于通信范围的5%，则认为已收敛
        if max_error < self.communication_range * 0.05:
            done['__all__'] = True

        return done

    def _get_info(self) -> Dict:
        """获取额外信息"""
        localization_errors = np.linalg.norm(
            self.estimated_positions - self.true_positions, axis=1
        )

        return {
            'mean_localization_error': np.mean(localization_errors),
            'max_localization_error': np.max(localization_errors),
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

            # 使用边界设置坐标轴范围
            x_min, x_max = self.initial_pos_bounds[0]
            y_min, y_max = self.initial_pos_bounds[1]

            # 添加一些边距以便更好地观察
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


class MultiAgentWrapper:
    """
    多智能体环境包装器，兼容主流MARL框架
    """

    def __init__(self, env):
        self.env = env
        self.agents = [f'sensor_{i}' for i in range(env.n_sensors)]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        return self.env.reset(seed=seed, options=options)

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        return self.env.step(actions)

    def render(self, mode: str = 'human') -> Any:
        return self.env.render(mode)

    def close(self) -> None:
        return self.env.close()

    @property
    def observation_space(self) -> spaces.Dict:
        return self.env.observation_space

    @property
    def action_space(self) -> spaces.Dict:
        return self.env.action_space


# 使用示例
if __name__ == "__main__":
    # 示例：定义锚点和传感器位置
    np.random.seed(42)
    anchors_pos = np.array([[0.0, 0.0], [250.0, 0.0], [0.0, 200.0], [250.0, 200.0]])
    sensor_pos = np.array([
        [50.0, 50.0], [100.0, 80.0], [150.0, 50.0], [200.0, 100.0],
        [50.0, 150.0], [100.0, 120.0], [150.0, 150.0], [200.0, 180.0]
    ])

    # 初始估计位置（添加噪声）
    estimated_positions = sensor_pos + np.random.normal(0, 15, sensor_pos.shape)

    # 创建环境
    env = SensorNetworkLocalizationEnv(
        anchors_pos=anchors_pos,
        sensor_pos=sensor_pos,
        estimated_positions=estimated_positions,
        communication_range=80.0,
        noise_std=1.5,
        initial_pos_bounds=np.array([[-50.0, 300.0], [-50.0, 250.0]]),
        reward_scale=0.1,
        global_reward_freq=3
    )

    # 包装为多智能体环境
    multi_env = MultiAgentWrapper(env)

    # 测试环境
    obs = multi_env.reset()
    print("观测空间维度:", {k: v.shape for k, v in list(obs.items())[:3]})  # 只显示前3个
    print("动作空间:", {k: v for k, v in list(multi_env.action_space.spaces.items())[:3]})

    # 测试MADDPG需要的功能
    print(f"全局状态维度: {env.get_global_state_dim()}")
    print(f"联合动作空间大小: {env.get_joint_action_space_size()}")
    print(f"单智能体观测维度: {env.get_local_observation_dim()}")

    global_state = env.get_global_state()
    print(f"当前全局状态形状: {global_state.shape}")

    # 随机动作测试
    for step in range(5):
        actions = {}
        for agent in multi_env.agents:
            actions[agent] = multi_env.action_space[agent].sample()

        obs, rewards, done, info = multi_env.step(actions)

        # 测试全局动作转换
        global_action = env.get_global_action_from_dict(actions)
        print(f"Step {step + 1}:")
        print(f"  平均定位误差: {info['mean_localization_error']:.4f}")
        print(f"  前3个智能体奖励: {[(k, f'{v:.4f}') for k, v in list(rewards.items())[:3]]}")
        print(f"  全局动作形状: {global_action.shape}")

        # 渲染环境
        multi_env.render()

        if done['__all__']:
            print("Episode finished early due to convergence!")
            break