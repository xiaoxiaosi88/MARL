# MASAC 专用配置
import numpy as np

# 环境配置
ANCHORS_POS = np.array([[-300, -300], [300, -300], [-300, 300], [300, 300]], dtype=np.float32)
SENSORS_POS = np.array([[0, 0], [150, 0], [-150, 0], [0, 150], [0, -150]], dtype=np.float32)
ESTIMATED_POSITIONS = np.array([[0, 0], [20, 0], [-10, 0], [90, 80], [50, -150]], dtype=np.float32)
COMMUNICATION_RANGE = 250
INITIAL_POS_BOUNDS_IVA = np.array([[-300.0, 300.0], [-300.0, 300.0]], dtype=np.float32)
NOISE_STD = 5.0  # 修改：增大到更合理的值
MAX_EPISODE_STEPS = 100
DIMENSION = 2
RENDER_MODE = None

# MASAC 算法配置
ACTOR_LR = 1e-4              # Actor学习率
CRITIC_LR = 1e-4             # Critic学习率
# 添加兼容性参数名
ACTOR_LEARNING_RATE = 1e-4   # 与 ACTOR_LR 保持一致
CRITIC_LEARNING_RATE = 1e-4  # 与 CRITIC_LR 保持一致

ALPHA_LR = 3e-4              # 熵系数学习率
HIDDEN_UNITS = [256, 256]    # 隐藏层单元数
GAMMA = 0.99                 # 折扣因子
TAU = 0.005                  # 软更新系数
ALPHA = 0.2                  # 初始熵系数
AUTO_ENTROPY = True          # 自动调整熵系数
TARGET_ENTROPY = -2          # 目标熵值

# 训练配置
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 256
LEARNING_STARTS = 5000       # 增加预热步数
POLICY_FREQUENCY = 1         # 策略更新频率
TARGET_NETWORK_FREQUENCY = 1
TOTAL_TIMESTEPS = 5000000    # 增加训练步数

# 探索配置
EXPLORATION_NOISE = 0.1
MAX_ACTION = 300.0
MIN_ACTION = -300.0
# 添加噪声相关参数
NOISE_SCALE = 0.2            # 噪声缩放因子
NOISE_DECAY = 0.995          # 噪声衰减率
MIN_NOISE = 0.01             # 最小噪声

# 奖励配置（新增）
REWARD_SCALE = 0.1           # 奖励整体缩放因子

# 多智能体特定配置
CENTRALIZED_TRAINING = True    # 中心化训练
DECENTRALIZED_EXECUTION = True # 去中心化执行
SHARE_PARAMETERS = False       # 是否共享参数
USE_GLOBAL_STATE = True        # 是否使用全局状态

# 实验配置
SEED = 42
EXPERIMENT_NAME = "sensor_network_masac"
LOG_DIR = "logs"
SAVE_MODEL = True
EVAL_FREQ = 10000             # 评估频率
SAVE_FREQ = 50000             # 保存频率
LOG_FREQ = 1000               # 日志频率

# 环境并行配置
NUM_ENVS = 8                  # 并行环境数量