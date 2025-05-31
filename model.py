import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


def init_weights(m, gain=0.01):
    """初始化网络权重"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class AttentionBlock(nn.Module):
    """注意力机制模块"""

    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.output(out)

        return self.norm(out + x)


class SpatialEncoder(nn.Module):
    """空间信息编码器"""

    def __init__(self, pos_dim=2, hidden_dim=128):
        super(SpatialEncoder, self).__init__()
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim

        # 位置编码
        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # 距离编码
        self.dist_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )

        self.fusion = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)

    def forward(self, pos, dist):
        pos_feat = self.pos_encoder(pos)
        dist_feat = self.dist_encoder(dist.unsqueeze(-1))

        combined = torch.cat([pos_feat, dist_feat], dim=-1)
        return F.relu(self.fusion(combined))


class OptimizedActor(nn.Module):
    """优化的Actor网络"""

    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256],
                 use_attention=True, use_spatial_encoding=True):
        super(OptimizedActor, self).__init__()

        self.use_attention = use_attention
        self.use_spatial_encoding = use_spatial_encoding

        # 解析观测维度
        self.pos_dim = 2
        # 根据观测维度估算邻居和锚点信息
        remaining_dim = obs_dim - self.pos_dim - self.pos_dim  # 减去当前位置和运动历史
        neighbor_anchor_dim = remaining_dim
        # 假设邻居和锚点信息各占一半
        self.max_neighbors = 4  # 简化处理
        self.max_anchors = 4

        if use_spatial_encoding:
            # 空间编码器
            self.spatial_encoder = SpatialEncoder(pos_dim=2, hidden_dim=128)
            spatial_feat_dim = 128
        else:
            spatial_feat_dim = 3  # 相对位置(2) + 距离(1)

        # 自身位置编码
        self.self_pos_encoder = nn.Sequential(
            nn.Linear(self.pos_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # 运动历史编码
        self.motion_encoder = nn.Sequential(
            nn.Linear(self.pos_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # 特征融合层
        if use_attention:
            # 注意力机制处理
            self.neighbor_attention = AttentionBlock(spatial_feat_dim, 128, num_heads=4)
            self.anchor_attention = AttentionBlock(spatial_feat_dim, 128, num_heads=4)
            attention_out_dim = 128 * 2
        else:
            # 简单的全连接处理
            neighbor_dim = neighbor_anchor_dim // 2
            anchor_dim = neighbor_anchor_dim // 2
            self.neighbor_fc = nn.Sequential(
                nn.Linear(neighbor_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            )
            self.anchor_fc = nn.Sequential(
                nn.Linear(anchor_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128)
            )
            attention_out_dim = 128 * 2

        # 主干网络
        total_input_dim = 64 + attention_out_dim + 32  # 自身位置 + 处理后的信息 + 运动历史

        layers = []
        prev_dim = total_input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # 输出层
        self.mu = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _parse_observation(self, obs):
        """解析观测"""
        batch_size = obs.size(0)

        # 当前位置 (2)
        self_pos = obs[:, :self.pos_dim]

        # 运动历史 (最后2维)
        motion_history = obs[:, -self.pos_dim:]

        # 中间部分包含邻居和锚点信息
        middle_part = obs[:, self.pos_dim:-self.pos_dim]

        # 简化处理：分别取前一半和后一半作为邻居和锚点信息
        mid_point = middle_part.size(1) // 2
        neighbor_info = middle_part[:, :mid_point]
        anchor_info = middle_part[:, mid_point:]

        return self_pos, neighbor_info, anchor_info, motion_history

    def forward(self, obs, deterministic=False):
        batch_size = obs.size(0)

        # 解析观测
        self_pos, neighbor_info, anchor_info, motion_history = self._parse_observation(obs)

        # 编码自身位置
        self_pos_feat = self.self_pos_encoder(self_pos)

        # 处理邻居信息
        if self.use_attention and neighbor_info.size(1) > 0:
            # 重新整形为序列格式以使用注意力
            neighbor_feat = neighbor_info.view(batch_size, -1, 3)  # 假设每个邻居3个特征
            neighbor_feat = self.neighbor_attention(neighbor_feat)
            neighbor_feat = neighbor_feat.mean(dim=1)  # 平均池化
        else:
            neighbor_feat = self.neighbor_fc(neighbor_info)

        # 处理锚点信息
        if self.use_attention and anchor_info.size(1) > 0:
            anchor_feat = anchor_info.view(batch_size, -1, 3)  # 假设每个锚点3个特征
            anchor_feat = self.anchor_attention(anchor_feat)
            anchor_feat = anchor_feat.mean(dim=1)  # 平均池化
        else:
            anchor_feat = self.anchor_fc(anchor_info)

        # 编码运动历史
        motion_feat = self.motion_encoder(motion_history)

        # 融合特征
        combined_feat = torch.cat([self_pos_feat, neighbor_feat, anchor_feat, motion_feat], dim=-1)

        # 主干网络
        features = self.backbone(combined_feat)

        # 输出动作
        mu = torch.tanh(self.mu(features))
        std = torch.exp(self.log_std.expand_as(mu))

        if deterministic:
            action = mu
            action_log_prob = None
        else:
            dist = Normal(mu, std)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, action_log_prob, mu, std

    def evaluate_actions(self, obs, actions):
        """评估动作的对数概率和熵"""
        _, _, mu, std = self.forward(obs, deterministic=False)

        dist = Normal(mu, std)
        action_log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return action_log_prob, entropy


class OptimizedCritic(nn.Module):
    """优化的Critic网络"""

    def __init__(self, input_dim, hidden_dims=[256, 256], use_dueling=True):
        super(OptimizedCritic, self).__init__()

        self.use_dueling = use_dueling

        # 共享特征提取器
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        if use_dueling:
            # Dueling架构
            self.value_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1] // 2, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1] // 2, 1)
            )
        else:
            # 标准架构
            self.value_head = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1], 1)
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shared_feat = self.shared_layers(x)

        if self.use_dueling:
            value = self.value_stream(shared_feat)
            advantage = self.advantage_stream(shared_feat)
            q_value = value + advantage - advantage.mean(dim=-1, keepdim=True)
            return q_value
        else:
            return self.value_head(shared_feat)


class OptimizedMAPPOPolicy(nn.Module):
    """优化的MAPPO策略网络"""

    def __init__(self, obs_dim, action_dim, state_dim=None,
                 hidden_dims=[256, 256], use_centralized_v=True,
                 use_attention=True, use_spatial_encoding=True, use_dueling=True):
        super(OptimizedMAPPOPolicy, self).__init__()

        self.use_centralized_v = use_centralized_v

        # Actor网络（分布式）
        self.actor = OptimizedActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            use_attention=use_attention,
            use_spatial_encoding=use_spatial_encoding
        )

        # Critic网络（中心化或分布式）
        critic_input_dim = state_dim if (use_centralized_v and state_dim) else obs_dim
        self.critic = OptimizedCritic(
            input_dim=critic_input_dim,
            hidden_dims=hidden_dims,
            use_dueling=use_dueling
        )

    def get_actions(self, obs, rnn_states_actor=None, masks=None, deterministic=False):
        actions, action_log_probs, mu, std = self.actor(obs, deterministic)
        return actions, action_log_probs, rnn_states_actor

    def get_values(self, state, rnn_states_critic=None, masks=None):
        values = self.critic(state)
        return values, rnn_states_critic

    def evaluate_actions(self, obs, state, actions, rnn_states_actor=None,
                         rnn_states_critic=None, masks=None):
        # 评估动作
        action_log_probs, entropy = self.actor.evaluate_actions(obs, actions)

        # 评估状态价值
        values = self.critic(state)

        return values, action_log_probs, entropy