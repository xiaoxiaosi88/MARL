import numpy as np
import functools

from pettingzoo import ParallelEnv
import networkx as nx
from typing import Optional, List, Dict, Tuple
from numpy.random import default_rng, SeedSequence
from gymnasium.spaces import Box, Discrete


class MyNewMultiAgentEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "localization_env_v0"}

    def __init__(self,
                 anchors_pos: Optional[List[List[float]]] = None,
                 sensor_pos: Optional[List[List[float]]] = None,
                 estimated_positions: Optional[List[List[float]]] = None,
                 communication_range: float = 250,
                 sensor_noise_std: float = 5.0,
                 anchor_noise_std: float = 3.0,
                 dimension: int = 2,
                 convergence_threshold: float = 2,
                 max_episode_steps: int = 100,
                 action_magnitude: float = 1.0,  # 动作幅值
                 initial_pos_bounds: Optional[np.ndarray] = None,  # 运动范围
                 boundary_penalty: float = -100.0,  # 边界惩罚
                 boundary_enforcement: str = "penalty",  # 边界处理方式: "penalty" 或 "clip"

                 **kwargs  # 添加kwargs以兼容BenchMARL的额外参数
                 ):
        super().__init__()
        # 处理配置参数，支持从列表转换为numpy数组
        if anchors_pos is None:
            anchors_pos = [[-500, 0], [0, -500], [500, 0], [0, 500]]
        if sensor_pos is None:
            sensor_pos = [ [150, 0], [-150, 0], [0, 150], [0, -150], [300, 300], [-300, 300], [300, -300],
                          [-300, -300]]
        if estimated_positions is None:
            estimated_positions = [[20, 0], [-10, 0], [90, 80], [50, -150], [10, 10], [100, -100], [10, 0],
                                   [20, 80]]
        if initial_pos_bounds is None:
            initial_pos_bounds = np.array([[-300.0, 300.0], [-300.0, 300.0]])

        # 转换为numpy数组
        self.anchors_pos = np.array(anchors_pos, dtype=np.float32)
        self.sensor_pos = np.array(sensor_pos, dtype=np.float32)
        self.initial_estimated_positions = np.array(estimated_positions, dtype=np.float32)
        self.initial_pos_bounds = np.array(initial_pos_bounds, dtype=np.float32)
        self.boundary_penalty = boundary_penalty
        self.boundary_enforcement = boundary_enforcement

        self.agent_num = self.sensor_pos.shape[0]
        self.possible_agents = [f"agent_{i}" for i in range(self.agent_num)]
        self.agents = self.possible_agents[:]
        self.anchors_num = self.anchors_pos.shape[0]
        self.dimension = dimension
        self.communication_range = communication_range
        self.max_episode_steps = max_episode_steps
        self.action_magnitude = action_magnitude

        # noise parameters
        self.sensor_noise_std = sensor_noise_std
        self.anchor_noise_std = anchor_noise_std

        # reward weights and thresholds
        self.convergence_threshold = convergence_threshold

        self._current_step = 0

        # 添加 PettingZoo 需要的属性
        self.max_neighbors = self.agent_num - 1  # 最大邻居数

        # 定义固定的锚点-传感器连接映射
        self.anchor_sensor_mapping = self._create_anchor_sensor_mapping()

        # 更新最大锚点数量（基于固定映射）
        self.max_anchors_per_sensor = max(len(anchors) for anchors in self.anchor_sensor_mapping.values())

        # 定义离散动作映射
        # 动作编号对应: 0=不动, 1=上, 2=下, 3=左, 4=右, 5=左上, 6=右上, 7=左下, 8=右下
        self.action_map = {
            0: np.array([0.0, 0.0]),  # 不动
            1: np.array([0.0, 1.0]),  # 上
            2: np.array([0.0, -1.0]),  # 下
            3: np.array([-1.0, 0.0]),  # 左
            4: np.array([1.0, 0.0]),  # 右
            5: np.array([-1.0, 1.0]),  # 左上
            6: np.array([1.0, 1.0]),  # 右上
            7: np.array([-1.0, -1.0]),  # 左下
            8: np.array([1.0, -1.0]),  # 右下
        }

        # build observation and action spaces
        self.obs_dim = self._calculate_obs_dim()
        self._observation_space = {
            a: Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_dim,),
                dtype=np.float32,
            )
            for a in self.possible_agents
        }
        # 修改为离散动作空间
        self._action_space = {
            b: Discrete(9)  # 9个离散动作
            for b in self.possible_agents
        }

        # placeholders for runtime
        self.true_positions = None
        self.anchor_positions = None
        self.estimated_positions = None
        self.communication_graph = None
        self.distance_measurements = None
        self.anchor_measurements = None
        self.np_random = None
        self.seed_value = None
        self._root_seed_seq: Optional[SeedSequence] = None
        self._last_actions = {}  # 存储每个智能体的原始动作
        self._boundary_violations = {}  # 存储边界违反信息

    def _create_anchor_sensor_mapping(self) -> Dict[int, List[int]]:
        """创建固定的锚点-传感器连接映射"""
        mapping = {}

        # 根据位置找到对应的传感器索引
        def find_sensor_index(target_pos):
            for i, pos in enumerate(self.sensor_pos):
                if np.allclose(pos, target_pos, atol=1e-6):
                    return i
            return None

        # 根据位置找到对应的锚点索引
        def find_anchor_index(target_pos):
            for i, pos in enumerate(self.anchors_pos):
                if np.allclose(pos, target_pos, atol=1e-6):
                    return i
            return None

        # 定义固定连接规则
        connections = [
            ([-500, 0], [-150, 0]),  # 锚点0 连接 传感器2
            ([0, -500], [0, -150]),  # 锚点1 连接 传感器4
            ([500, 0], [150, 0]),  # 锚点2 连接 传感器1
            ([0, 500], [0, 150]),  # 锚点3 连接 传感器3
        ]

        # 初始化映射
        for i in range(self.agent_num):
            mapping[i] = []

        # 建立连接
        for anchor_pos, sensor_pos in connections:
            anchor_idx = find_anchor_index(anchor_pos)
            sensor_idx = find_sensor_index(sensor_pos)

            if anchor_idx is not None and sensor_idx is not None:
                mapping[sensor_idx].append(anchor_idx)
                print(f"Fixed connection: Anchor {anchor_idx} {anchor_pos} <-> Sensor {sensor_idx} {sensor_pos}")
            else:
                print(f"Warning: Could not find indices for connection {anchor_pos} <-> {sensor_pos}")

        return mapping

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """返回指定智能体的观察空间"""
        return self._observation_space[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """返回指定智能体的动作空间"""
        return self._action_space[agent]

    def _discrete_action_to_continuous(self, action: int) -> np.ndarray:
        """将离散动作转换为连续动作"""
        if action not in self.action_map:
            raise ValueError(f"Invalid action: {action}. Must be in range [0, 8]")

        # 获取基础方向向量并乘以动作幅值
        direction = self.action_map[action]
        return direction * self.action_magnitude

    def _is_position_in_bounds(self, position: np.ndarray) -> bool:
        """检查位置是否在边界内"""
        return bool((position[0] >= self.initial_pos_bounds[0, 0] and
                     position[0] <= self.initial_pos_bounds[0, 1] and
                     position[1] >= self.initial_pos_bounds[1, 0] and
                     position[1] <= self.initial_pos_bounds[1, 1]))

    def _clip_position_to_bounds(self, position: np.ndarray) -> np.ndarray:
        """将位置限制在边界内"""
        clipped_pos = position.copy()
        clipped_pos[0] = np.clip(clipped_pos[0],
                                 self.initial_pos_bounds[0, 0],
                                 self.initial_pos_bounds[0, 1])
        clipped_pos[1] = np.clip(clipped_pos[1],
                                 self.initial_pos_bounds[1, 0],
                                 self.initial_pos_bounds[1, 1])
        return clipped_pos

    def _get_boundary_penalty(self, agent_id: int, new_position: np.ndarray) -> float:
        """计算边界惩罚"""
        if self._is_position_in_bounds(new_position):
            return 0.0

        # 计算距离边界的距离
        x_violation = np.maximum(0, self.initial_pos_bounds[0, 0] - new_position[0]) + \
                      np.maximum(0, new_position[0] - self.initial_pos_bounds[0, 1])
        y_violation = np.maximum(0, self.initial_pos_bounds[1, 0] - new_position[1]) + \
                      np.maximum(0, new_position[1] - self.initial_pos_bounds[1, 1])

        # 基础惩罚加上距离相关的额外惩罚
        distance_penalty = np.sqrt(x_violation ** 2 + y_violation ** 2)
        return self.boundary_penalty - distance_penalty

    @property
    def max_num_agents(self):
        return self.agent_num

    @property
    def num_agents(self):
        return len(self.agents)

    def state(self):
        """返回全局状态"""
        if self.estimated_positions is None:
            return np.zeros(self.obs_dim * self.agent_num, dtype=np.float32)

        # global state = concatenated observations
        obs_list = [self._get_agent_observation(i) for i in range(self.agent_num)]
        return np.concatenate(obs_list).astype(np.float32)

    def reset(self, seed=None, options=None):
        """重置环境，正确处理种子"""
        # 处理种子
        if seed is not None:
            if self._root_seed_seq is None:
                self._root_seed_seq = SeedSequence(seed)
            else:
                # 更新种子序列
                self._root_seed_seq = SeedSequence(seed)
        elif self._root_seed_seq is None:
            self._root_seed_seq = SeedSequence()

        # 创建新的随机数生成器
        child_ss = self._root_seed_seq.spawn(1)[0]
        self.np_random = default_rng(child_ss)

        # reset positions and graph
        self._current_step = 0
        self.agents = self.possible_agents[:]
        self.true_positions = self.sensor_pos.copy()
        self.anchor_positions = self.anchors_pos.copy()
        self.estimated_positions = self.initial_estimated_positions.copy()
        self.communication_graph = self._build_communication_graph()
        self._generate_measurements()
        self._boundary_violations = {agent: False for agent in self.agents}

        # 确保初始位置在边界内
        for i in range(self.agent_num):
            if not self._is_position_in_bounds(self.estimated_positions[i]):
                self.estimated_positions[i] = self._clip_position_to_bounds(self.estimated_positions[i])
                print(f"Warning: Agent {i} initial position was out of bounds and has been clipped.")

        # build observation dict
        obs = {agent: self._get_agent_observation(idx)
               for idx, agent in enumerate(self.agents)}
        info = {agent: {} for agent in self.agents}
        return obs, info

    def step(self, action_dict):
        """执行一步，返回 PettingZoo 格式"""
        self._current_step += 1

        # 存储原始动作并转换为连续动作
        continuous_actions = {}
        for agent, action in action_dict.items():
            self._last_actions[agent] = action
            continuous_actions[agent] = self._discrete_action_to_continuous(action)

        # apply actions (相对移动) 并处理边界
        new_pos = self.estimated_positions.copy()
        boundary_penalties = {}

        for idx, agent in enumerate(self.agents):
            if agent in continuous_actions:
                # 计算新位置
                proposed_pos = self.estimated_positions[idx] + continuous_actions[agent]

                # 检查边界违反
                if not self._is_position_in_bounds(proposed_pos):
                    self._boundary_violations[agent] = True
                    boundary_penalties[agent] = self._get_boundary_penalty(idx, proposed_pos)

                    # 根据边界处理方式决定最终位置
                    if self.boundary_enforcement == "clip":
                        new_pos[idx] = self._clip_position_to_bounds(proposed_pos)
                    else:  # penalty
                        new_pos[idx] = proposed_pos  # 允许越界但给予惩罚
                else:
                    self._boundary_violations[agent] = False
                    boundary_penalties[agent] = 0.0
                    new_pos[idx] = proposed_pos
            else:
                boundary_penalties[agent] = 0.0
                self._boundary_violations[agent] = False

        self.estimated_positions = new_pos

        # compute losses
        global_loss = self._compute_global_loss()

        # rewards
        global_reward = -global_loss

        # done flags
        terminated = {agent: False for agent in self.agents}

        truncated = {agent: self._current_step > self.max_episode_steps for agent in self.agents}
        truncated_flag = self._check_global_convergence(global_loss)
        if truncated_flag:
            for agent in self.agents:
                truncated[agent] = True

        # prepare returns
        obs = {agent: self._get_agent_observation(idx)
               for idx, agent in enumerate(self.agents)}

        # 计算最终奖励：全局奖励 + 个体边界惩罚
        rewards = {}
        for agent in self.agents:
            base_reward = float(global_reward)
            boundary_penalty = boundary_penalties.get(agent, 0.0)
            rewards[agent] = base_reward + boundary_penalty

        info = {agent: {
            'localization_error': float(np.linalg.norm(self.estimated_positions[idx] - self.true_positions[idx])),
            'global_loss': float(global_loss),
            'last_action': self._last_actions.get(agent, 0),
            'action_description': self._get_action_description(self._last_actions.get(agent, 0)),
            'boundary_violation': self._boundary_violations.get(agent, False),
            'boundary_penalty': boundary_penalties.get(agent, 0.0),
            'position_x': float(self.estimated_positions[idx][0]),
            'position_y': float(self.estimated_positions[idx][1]),
            'in_bounds': self._is_position_in_bounds(self.estimated_positions[idx]),
            'connected_anchors': self.anchor_sensor_mapping[idx]
        } for idx, agent in enumerate(self.agents)}

        return obs, rewards, terminated, truncated, info

    def _get_action_description(self, action: int) -> str:
        """获取动作描述"""
        action_descriptions = {
            0: "stay",
            1: "up",
            2: "down",
            3: "left",
            4: "right",
            5: "left_up",
            6: "right_up",
            7: "left_down",
            8: "right_down"
        }
        return action_descriptions.get(action, "unknown")

    def render(self, mode="human"):
        """渲染环境"""
        if mode == "human":
            print(f"Step: {self._current_step}")
            print(f"Boundary: X[{self.initial_pos_bounds[0, 0]:.1f}, {self.initial_pos_bounds[0, 1]:.1f}], "
                  f"Y[{self.initial_pos_bounds[1, 0]:.1f}, {self.initial_pos_bounds[1, 1]:.1f}]")

            print("\nFixed Anchor-Sensor Connections:")
            for sensor_idx, anchor_indices in self.anchor_sensor_mapping.items():
                if anchor_indices:
                    anchor_info = []
                    for anchor_idx in anchor_indices:
                        anchor_pos = self.anchors_pos[anchor_idx]
                        anchor_info.append(f"Anchor{anchor_idx}{anchor_pos}")
                    print(f"  Sensor {sensor_idx} {self.sensor_pos[sensor_idx]} <-> {', '.join(anchor_info)}")

            print("\nAgent Status:")
            for idx, agent in enumerate(self.possible_agents):
                if self.estimated_positions is not None:
                    est_pos = self.estimated_positions[idx]
                    true_pos = self.true_positions[idx]
                    error = np.linalg.norm(est_pos - true_pos)
                    last_action = self._last_actions.get(agent, 0)
                    action_desc = self._get_action_description(last_action)
                    boundary_violation = self._boundary_violations.get(agent, False)
                    in_bounds = self._is_position_in_bounds(est_pos)
                    connected_anchors = self.anchor_sensor_mapping[idx]

                    boundary_status = "OUT" if not in_bounds else "IN"
                    violation_marker = "⚠️" if boundary_violation else "✅"

                    print(f"  {agent}: Est{est_pos}, True{true_pos}, Error{error:.2f}, "
                          f"Action:{action_desc}, Bounds:{boundary_status} {violation_marker}, "
                          f"Anchors:{connected_anchors}")

    def close(self):
        """关闭环境"""
        pass

    def _calculate_obs_dim(self):
        """计算新的观测空间维度"""
        obs_dim = 0

        # 1. 当前节点的估计位置 (x, y)
        obs_dim += self.dimension

        # 2. 邻居节点的当前位置 + 与邻居的距离测量值
        # 每个邻居：位置(x, y) + 距离测量值(1) = 3个特征
        neighbor_feature_dim = self.dimension + 1
        obs_dim += self.max_neighbors * neighbor_feature_dim

        # 3. 与锚节点的距离测量值（基于固定连接）
        # 每个锚点：距离测量值(1) = 1个特征
        anchor_feature_dim = 1
        obs_dim += self.max_anchors_per_sensor * anchor_feature_dim

        return obs_dim

    def _compute_global_loss(self) -> float:
        """计算全局损失函数 L(X)"""
        total_loss = 0.0

        # 传感器间距离损失
        for (i, j) in self.communication_graph.edges():
            measured_dist = self.distance_measurements[(i, j)]
            estimated_dist = np.linalg.norm(self.estimated_positions[i] - self.estimated_positions[j])
            residual = (measured_dist - estimated_dist) ** 2
            total_loss += residual / (self.sensor_noise_std ** 2) * 2

        # 传感器-锚点距离损失（基于固定连接）
        for (i, k), measured_dist in self.anchor_measurements.items():
            estimated_dist = np.linalg.norm(self.estimated_positions[i] - self.anchor_positions[k])
            residual = (measured_dist - estimated_dist) ** 2
            total_loss += residual / (self.anchor_noise_std ** 2) * 2

        return total_loss

    def _generate_measurements(self):
        """生成带有噪声的距离测量"""
        self.distance_measurements = {}
        for i, j in self.communication_graph.edges():
            true_distance = np.linalg.norm(self.true_positions[i] - self.true_positions[j])
            noise = self.np_random.normal(0, self.sensor_noise_std)
            self.distance_measurements[(i, j)] = max(0.1, true_distance + noise)  # 避免负距离

        # 基于固定连接生成锚点测量
        self.anchor_measurements = {}
        for sensor_idx, anchor_indices in self.anchor_sensor_mapping.items():
            for anchor_idx in anchor_indices:
                # 计算真实距离
                true_distance = np.linalg.norm(self.true_positions[sensor_idx] - self.anchor_positions[anchor_idx])
                # 添加噪声
                noise = self.np_random.normal(0, self.anchor_noise_std)
                self.anchor_measurements[(sensor_idx, anchor_idx)] = max(0.1, true_distance + noise)

    def _build_communication_graph(self):
        """构建通信图"""
        G = nx.Graph()
        G.add_nodes_from(range(self.agent_num))
        for i in range(self.agent_num):
            for j in range(i + 1, self.agent_num):
                if np.linalg.norm(self.true_positions[i] - self.true_positions[j]) <= self.communication_range:
                    G.add_edge(i, j)
        return G

    def _get_agent_observation(self, agent_id: int) -> np.ndarray:
        """获取智能体的新观测"""
        obs_components = []

        # 1. 当前节点的估计位置
        current_pos = self.estimated_positions[agent_id]
        obs_components.append(current_pos)

        # 2. 邻居节点信息：位置 + 距离测量值
        neighbors = list(self.communication_graph.neighbors(agent_id))
        neighbor_feature_dim = self.dimension + 1  # 位置(x,y) + 距离测量值

        neighbor_data = np.zeros(self.max_neighbors * neighbor_feature_dim, dtype=np.float32)

        for i, neighbor in enumerate(neighbors):
            if i >= self.max_neighbors:
                break

            # 邻居的当前位置
            neighbor_pos = self.estimated_positions[neighbor]

            # 与邻居的距离测量值
            measurement_key = (agent_id, neighbor) if (agent_id, neighbor) in self.distance_measurements else (
                neighbor, agent_id)
            measured_distance = self.distance_measurements.get(measurement_key, 0.0)

            start_idx = i * neighbor_feature_dim

            # 邻居位置 (x, y)
            neighbor_data[start_idx:start_idx + self.dimension] = neighbor_pos
            # 距离测量值
            neighbor_data[start_idx + self.dimension] = measured_distance

        obs_components.append(neighbor_data)

        # 3. 与锚节点的距离测量值（基于固定连接）
        anchor_feature_dim = 1  # 只有距离测量值
        anchor_data = np.zeros(self.max_anchors_per_sensor * anchor_feature_dim, dtype=np.float32)

        # 获取该传感器连接的锚点
        connected_anchors = self.anchor_sensor_mapping[agent_id]

        for i, anchor_idx in enumerate(connected_anchors):
            if i >= self.max_anchors_per_sensor:
                break

            # 获取与该锚点的距离测量值
            measured_distance = self.anchor_measurements.get((agent_id, anchor_idx), 0.0)

            start_idx = i * anchor_feature_dim
            anchor_data[start_idx] = measured_distance

        obs_components.append(anchor_data)

        # 组合观测
        observation = np.concatenate(obs_components).astype(np.float32)
        return observation

    def _check_global_convergence(self, current_loss: float) -> bool:
        errors = np.linalg.norm(self.estimated_positions - self.true_positions, axis=1)
        return bool(np.all(errors < self.convergence_threshold))

    # 辅助方法
    def get_estimated_positions(self) -> np.ndarray:
        """获取当前估计位置"""
        return self.estimated_positions.copy()

    def get_true_positions(self) -> np.ndarray:
        """获取真实位置"""
        return self.true_positions.copy()

    def get_position_errors(self) -> np.ndarray:
        """获取位置误差"""
        return np.linalg.norm(self.estimated_positions - self.true_positions, axis=1)

    def get_global_loss(self) -> float:
        """获取当前全局损失"""
        return self._compute_global_loss()

    def get_action_meanings(self) -> List[str]:
        """获取动作含义列表"""
        return ["stay", "up", "down", "left", "right", "left_up", "right_up", "left_down", "right_down"]

    def get_boundary_info(self) -> dict:
        """获取边界信息"""
        return {
            'bounds': self.initial_pos_bounds.copy(),
            'boundary_penalty': self.boundary_penalty,
            'boundary_enforcement': self.boundary_enforcement,
            'agents_in_bounds': {agent: self._is_position_in_bounds(self.estimated_positions[idx])
                                 for idx, agent in enumerate(self.agents)},
            'boundary_violations': self._boundary_violations.copy()
        }

    def get_agents_out_of_bounds(self) -> List[str]:
        """获取越界的智能体列表"""
        out_of_bounds = []
        for idx, agent in enumerate(self.agents):
            if not self._is_position_in_bounds(self.estimated_positions[idx]):
                out_of_bounds.append(agent)
        return out_of_bounds

    def get_observation_info(self) -> dict:
        """获取观测空间信息"""
        return {
            'obs_dim': self.obs_dim,
            'current_position_dim': self.dimension,
            'neighbor_feature_dim': self.dimension + 1,
            'max_neighbors': self.max_neighbors,
            'anchor_feature_dim': 1,
            'max_anchors_per_sensor': self.max_anchors_per_sensor,
            'anchor_sensor_mapping': self.anchor_sensor_mapping,
            'observation_structure': {
                'current_position': f'[0:{self.dimension}]',
                'neighbor_info': f'[{self.dimension}:{self.dimension + self.max_neighbors * (self.dimension + 1)}]',
                'anchor_distances': f'[{self.dimension + self.max_neighbors * (self.dimension + 1)}:{self.obs_dim}]'
            }
        }

    def get_anchor_sensor_connections(self) -> Dict[int, List[int]]:
        """获取锚点-传感器连接信息"""
        return self.anchor_sensor_mapping.copy()


# 添加环境工厂函数
def parallel_env(**kwargs):
    """环境工厂函数"""
    return MyNewMultiAgentEnv(**kwargs)