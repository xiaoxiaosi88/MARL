import torch
import numpy as np
import random
import argparse
import os
from datetime import datetime

# 导入自定义模块
from MRL_ENV import SensorNetworkLocalizationEnv, MAPPOWrapper
from trainer import MAPPOTrainer
import config_mappo


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_env(config):
    """创建环境"""
    env = SensorNetworkLocalizationEnv(
        anchors_pos=config['ANCHORS_POS'],
        sensor_pos=config['SENSORS_POS'],
        estimated_positions=config['ESTIMATED_POSITIONS'],
        communication_range=config['COMMUNICATION_RANGE'],
        noise_std=config['NOISE_STD'],
        max_episode_steps=config['MAX_EPISODE_STEPS'],
        initial_pos_bounds=config['INITIAL_POS_BOUNDS_IVA'],
        render_mode=config['RENDER_MODE'],
        dimension=config['DIMENSION'],
        reward_scale=1.0,
        global_reward_freq=5,
        use_centralized_critic=config['USE_CENTRALIZED_V']
    )

    wrapped_env = MAPPOWrapper(env)
    return wrapped_env


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MAPPO for Sensor Network Localization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    parser.add_argument('--eval', action='store_true', default=False, help='Evaluation mode')
    parser.add_argument('--model_path', type=str, default=None, help='Path to load model')
    parser.add_argument('--render', action='store_true', default=False, help='Render environment')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建配置字典
    config_dict = {
        # 环境配置
        'ANCHORS_POS': config_mappo.ANCHORS_POS,
        'SENSORS_POS': config_mappo.SENSORS_POS,
        'ESTIMATED_POSITIONS': config_mappo.ESTIMATED_POSITIONS,
        'COMMUNICATION_RANGE': config_mappo.COMMUNICATION_RANGE,
        'INITIAL_POS_BOUNDS_IVA': config_mappo.INITIAL_POS_BOUNDS_IVA,
        'NOISE_STD': config_mappo.NOISE_STD,
        'MAX_EPISODE_STEPS': config_mappo.MAX_EPISODE_STEPS,
        'DIMENSION': config_mappo.DIMENSION,
        'RENDER_MODE': config_mappo.RENDER_MODE,

        # 网络配置
        'HIDDEN_UNITS': config_mappo.HIDDEN_UNITS,
        'USE_ATTENTION': config_mappo.USE_ATTENTION,
        'USE_SPATIAL_ENCODING': config_mappo.USE_SPATIAL_ENCODING,
        'USE_DUELING': config_mappo.USE_DUELING,
        'USE_CENTRALIZED_V': config_mappo.USE_CENTRALIZED_V,

        # MAPPO算法配置
        'ACTOR_LR': config_mappo.ACTOR_LR,
        'CRITIC_LR': config_mappo.CRITIC_LR,
        'GAMMA': config_mappo.GAMMA,
        'GAE_LAMBDA': config_mappo.GAE_LAMBDA,
        'EPS_CLIP': config_mappo.EPS_CLIP,
        'VALUE_LOSS_COEF': config_mappo.VALUE_LOSS_COEF,
        'ENTROPY_COEF': config_mappo.ENTROPY_COEF,
        'MAX_GRAD_NORM': config_mappo.MAX_GRAD_NORM,
        'PPO_EPOCHS': config_mappo.PPO_EPOCHS,
        'NUM_MINI_BATCH': config_mappo.NUM_MINI_BATCH,
        'USE_VALUE_ACTIVE_MASKS': config_mappo.USE_VALUE_ACTIVE_MASKS,
        'USE_POLICY_ACTIVE_MASKS': config_mappo.USE_POLICY_ACTIVE_MASKS,

        # 训练配置
        'ROLLOUT_LENGTH': config_mappo.ROLLOUT_LENGTH,
        'BATCH_SIZE': config_mappo.BATCH_SIZE,
        'TOTAL_TIMESTEPS': config_mappo.TOTAL_TIMESTEPS,
        'EVAL_EPISODES': config_mappo.EVAL_EPISODES,

        # 优化配置
        'USE_LINEAR_LR_DECAY': config_mappo.USE_LINEAR_LR_DECAY,
        'USE_FEATURE_NORMALIZATION': config_mappo.USE_FEATURE_NORMALIZATION,
        'USE_HUBER_LOSS': config_mappo.USE_HUBER_LOSS,
        'HUBER_DELTA': config_mappo.HUBER_DELTA,
        'USE_CLIPPED_VALUE_LOSS': config_mappo.USE_CLIPPED_VALUE_LOSS,
        'USE_OBS_INSTEAD_OF_STATE': config_mappo.USE_OBS_INSTEAD_OF_STATE,

        # 自适应训练配置
        'USE_ADAPTIVE_HYPERPARAMETERS': config_mappo.USE_ADAPTIVE_HYPERPARAMETERS,
        'USE_PHASED_TRAINING': config_mappo.USE_PHASED_TRAINING,
        'TARGET_KL_DIVERGENCE': config_mappo.TARGET_KL_DIVERGENCE,
        'ENTROPY_ADAPTATION_RATE': config_mappo.ENTROPY_ADAPTATION_RATE,
        'LR_DECAY_FACTOR': config_mappo.LR_DECAY_FACTOR,
        'EARLY_STOP_PATIENCE': config_mappo.EARLY_STOP_PATIENCE,
        'EARLY_STOP_THRESHOLD': config_mappo.EARLY_STOP_THRESHOLD,
        'TRAINING_PHASES': config_mappo.TRAINING_PHASES,

        # 实验配置
        'SEED': args.seed,
        'SAVE_FREQ': config_mappo.SAVE_FREQ,
        'EVAL_FREQ': config_mappo.EVAL_FREQ,
        'LOG_FREQ': config_mappo.LOG_FREQ,
        'NUM_ENVS': config_mappo.NUM_ENVS,
    }

    # 设置日志目录
    if args.log_dir:
        log_dir = args.log_dir
    else:
        experiment_name = args.experiment_name or config_mappo.EXPERIMENT_NAME
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(config_mappo.LOG_DIR, f"{experiment_name}_{timestamp}")

    config_dict['LOG_DIR'] = log_dir
    os.makedirs(log_dir, exist_ok=True)

    # 创建环境
    env = create_env(config_dict)

    print(f"Environment created with {env.get_num_agents()} agents")
    print(f"Observation dim: {env.get_obs_dim()}")
    print(f"Action dim: {env.get_action_dim()}")
    print(f"State dim: {env.get_state_dim()}")

    if args.eval:
        # 评估模式
        if args.model_path is None:
            raise ValueError("Model path is required for evaluation")

        trainer = MAPPOTrainer(env, config_dict)
        trainer.algorithm.load(args.model_path)
        print(f"Model loaded from {args.model_path}")

        eval_info = trainer.evaluate()
        print(f"Evaluation results:")
        for key, value in eval_info.items():
            print(f"  {key}: {value:.4f}")

        if args.render:
            for episode in range(5):
                obs, info = env.reset()
                episode_reward = 0
                done = False
                step = 0

                print(f"\nEpisode {episode + 1}:")

                while not done and step < config_mappo.MAX_EPISODE_STEPS:
                    obs_tensor = torch.from_numpy(
                        np.stack([obs[agent] for agent in env.get_agent_ids()])
                    ).float()

                    with torch.no_grad():
                        actions, _, _ = trainer.policy.get_actions(obs_tensor, deterministic=True)

                    actions_dict = {}
                    for i, agent in enumerate(env.get_agent_ids()):
                        actions_dict[agent] = actions[i].numpy()

                    obs, rewards, dones, truncated, info = env.step(actions_dict)
                    episode_reward += sum(rewards.values())
                    step += 1

                    done = dones.get('__all__', False) or truncated.get('__all__', False)

                    if step % 20 == 0:
                        env.render()

                print(f"  Episode reward: {episode_reward:.2f}")
                print(f"  Episode length: {step}")
                print(f"  Final localization error: {info.get('mean_localization_error', 0):.2f}")

    else:
        # 训练模式
        trainer = MAPPOTrainer(env, config_dict)

        print("🚀 Starting automatic MAPPO training...")
        print(f"📊 Configuration:")
        print(f"  - Total timesteps: {config_dict['TOTAL_TIMESTEPS']:,}")
        print(f"  - Rollout length: {config_dict['ROLLOUT_LENGTH']}")
        print(f"  - Batch size: {config_dict['BATCH_SIZE']}")
        print(f"  - Adaptive hyperparameters: {config_dict['USE_ADAPTIVE_HYPERPARAMETERS']}")
        print(f"  - Phased training: {config_dict['USE_PHASED_TRAINING']}")
        print(f"  - Log directory: {log_dir}")

        if config_dict['USE_PHASED_TRAINING']:
            print(f"📈 Training phases:")
            for phase_id, phase_info in config_dict['TRAINING_PHASES'].items():
                print(f"  Phase {phase_id + 1}: {phase_info['name']} ({phase_info['max_steps']:,} steps)")

        print("\n" + "=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)

        try:
            trainer.train()
            print("\n🎉 Training completed successfully!")
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
            trainer.save_model("interrupted_model.pt")
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            trainer.save_model("error_model.pt")
            raise


if __name__ == "__main__":
    main()