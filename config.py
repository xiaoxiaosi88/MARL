import numpy as np

# ============= 环境配置 =============
ANCHORS_POS = np.array([[-300, -300], [300, -300], [-300, 300], [300, 300]], dtype=np.float32)
SENSORS_POS = np.array([[0, 0], [150, 0], [-150, 0], [0, 150], [0, -150]], dtype=np.float32)
ESTIMATED_POSITIONS = np.array([[0, 0], [20, 0], [-10, 0], [90, 80], [50, -150]], dtype=np.float32)
COMMUNICATION_RANGE = 250
INITIAL_POS_BOUNDS_IVA = np.array([[-300.0, 300.0], [-300.0, 300.0]], dtype=np.float32)
NOISE_STD = 5.0
MAX_EPISODE_STEPS = 200
DIMENSION = 2
RENDER_MODE = None

# ============= 神经网络架构配置 =============
HIDDEN_UNITS = [256, 256, 128]
USE_ATTENTION = True
USE_SPATIAL_ENCODING = True
USE_DUELING = True
USE_CENTRALIZED_V = True

# ============= MAPPO算法配置 =============
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPS_CLIP = 0.2
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.02
MAX_GRAD_NORM = 0.5

# PPO特有参数
PPO_EPOCHS = 10
NUM_MINI_BATCH = 4
USE_VALUE_ACTIVE_MASKS = True
USE_POLICY_ACTIVE_MASKS = True

# ============= 训练配置 =============
ROLLOUT_LENGTH = 1024
BATCH_SIZE = 256
TOTAL_TIMESTEPS = 5000000
EVAL_EPISODES = 32

# ============= 优化策略配置 =============
USE_LINEAR_LR_DECAY = True
USE_FEATURE_NORMALIZATION = True
USE_HUBER_LOSS = True
HUBER_DELTA = 10.0
USE_CLIPPED_VALUE_LOSS = True
USE_OBS_INSTEAD_OF_STATE = False

# ============= 自适应训练配置 =============
USE_ADAPTIVE_HYPERPARAMETERS = True
USE_PHASED_TRAINING = True
TARGET_KL_DIVERGENCE = 0.02
ENTROPY_ADAPTATION_RATE = 0.02
LR_DECAY_FACTOR = 0.9
EARLY_STOP_PATIENCE = 10
EARLY_STOP_THRESHOLD = 1.0

# ============= 分阶段训练配置 =============
TRAINING_PHASES = {
    0: {
        'name': 'Basic_Navigation',
        'max_steps': 200000,
        'min_steps': 100000,
        'description': '学习基础移动和环境适应',
        'config_overrides': {
            'ENTROPY_COEF': 0.05,
            'ACTOR_LR': 5e-4,
            'EPS_CLIP': 0.3,
            'ROLLOUT_LENGTH': 512,
        },
        'success_criteria': {
            'min_episode_length': 50,
            'max_localization_error': 50.0,
        }
    },
    1: {
        'name': 'Local_Coordination',
        'max_steps': 500000,
        'min_steps': 200000,
        'description': '学习智能体间协调',
        'config_overrides': {
            'ENTROPY_COEF': 0.03,
            'ACTOR_LR': 3e-4,
            'EPS_CLIP': 0.2,
            'ROLLOUT_LENGTH': 1024,
        },
        'success_criteria': {
            'max_localization_error': 20.0,
            'min_coordination_score': 0.7,
        }
    },
    2: {
        'name': 'Fine_Tuning',
        'max_steps': 1000000,
        'min_steps': 300000,
        'description': '精细调优和全局优化',
        'config_overrides': {
            'ENTROPY_COEF': 0.01,
            'ACTOR_LR': 1e-4,
            'EPS_CLIP': 0.1,
            'ROLLOUT_LENGTH': 2048,
        },
        'success_criteria': {
            'max_localization_error': 5.0,
            'convergence_rate': 0.9,
        }
    }
}

# ============= 实验配置 =============
SEED = 42
EXPERIMENT_NAME = "sensor_network_mappo_optimized"
LOG_DIR = "logs"
SAVE_FREQ = 100000
EVAL_FREQ = 25000
LOG_FREQ = 1000
NUM_ENVS = 16
