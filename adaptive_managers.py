import numpy as np
import torch
import json
import os
from datetime import datetime


class HyperparameterManager:
    """超参数自适应管理器"""

    def __init__(self, config, trainer):
        self.config = config
        self.trainer = trainer
        self.step_count = 0

        # 初始参数
        self.initial_actor_lr = config['ACTOR_LR']
        self.initial_critic_lr = config.get('CRITIC_LR', config['ACTOR_LR'])
        self.initial_entropy_coef = config['ENTROPY_COEF']

        # 当前参数
        self.current_actor_lr = self.initial_actor_lr
        self.current_critic_lr = self.initial_critic_lr
        self.current_entropy_coef = self.initial_entropy_coef

        # 性能历史
        self.performance_history = []
        self.best_performance = float('inf')
        self.no_improvement_count = 0

        # 保存路径
        self.log_file = os.path.join(config['LOG_DIR'], 'hyperparameter_log.json')

    def update(self, step, train_metrics, eval_metrics=None):
        """更新超参数"""
        self.step_count = step

        # 1. 学习率线性衰减
        self._update_learning_rates()

        # 2. 根据KL散度调整熵系数
        if 'kl_divergence' in train_metrics:
            self._update_entropy_coefficient(train_metrics['kl_divergence'])

        # 3. 记录性能
        if eval_metrics:
            self._record_performance(eval_metrics)

        # 4. 应用更新
        self._apply_updates()

        # 5. 记录日志
        self._log_changes(train_metrics, eval_metrics)

    def _update_learning_rates(self):
        """线性衰减学习率"""
        total_steps = self.config['TOTAL_TIMESTEPS']
        progress = min(self.step_count / total_steps, 1.0)
        decay_factor = self.config.get('LR_DECAY_FACTOR', 0.9)

        self.current_actor_lr = self.initial_actor_lr * (1.0 - (1.0 - decay_factor) * progress)
        self.current_critic_lr = self.initial_critic_lr * (1.0 - (1.0 - decay_factor) * progress)

    def _update_entropy_coefficient(self, kl_divergence):
        """根据KL散度调整熵系数"""
        target_kl = self.config.get('TARGET_KL_DIVERGENCE', 0.02)
        adaptation_rate = self.config.get('ENTROPY_ADAPTATION_RATE', 0.02)

        if kl_divergence > target_kl * 1.5:
            # KL太大，减少探索
            self.current_entropy_coef *= (1.0 - adaptation_rate)
        elif kl_divergence < target_kl * 0.5:
            # KL太小，增加探索
            self.current_entropy_coef *= (1.0 + adaptation_rate)

        # 限制范围
        self.current_entropy_coef = np.clip(self.current_entropy_coef, 0.001, 0.1)

    def _record_performance(self, eval_metrics):
        """记录和分析性能"""
        current_error = eval_metrics.get('mean_localization_error', float('inf'))
        self.performance_history.append(current_error)

        if current_error < self.best_performance:
            self.best_performance = current_error
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

    def _apply_updates(self):
        """将更新应用到训练器"""
        # 更新优化器学习率
        for param_group in self.trainer.algorithm.optimizer.param_groups:
            param_group['lr'] = self.current_actor_lr

        # 更新熵系数
        self.trainer.algorithm.entropy_coef = self.current_entropy_coef

    def _log_changes(self, train_metrics, eval_metrics):
        """记录参数变化"""
        log_entry = {
            'step': self.step_count,
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': {
                'actor_lr': self.current_actor_lr,
                'critic_lr': self.current_critic_lr,
                'entropy_coef': self.current_entropy_coef,
            },
            'metrics': {
                'train': train_metrics,
                'eval': eval_metrics
            }
        }

        # 追加到日志文件
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)

        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def should_early_stop(self):
        """判断是否早停"""
        patience = self.config.get('EARLY_STOP_PATIENCE', 10)
        threshold = self.config.get('EARLY_STOP_THRESHOLD', 1.0)

        # 性能阈值检查
        if self.best_performance < threshold:
            return True

        # 无改善次数检查
        return self.no_improvement_count >= patience

    def get_current_params(self):
        """获取当前参数"""
        return {
            'actor_lr': self.current_actor_lr,
            'critic_lr': self.current_critic_lr,
            'entropy_coef': self.current_entropy_coef,
        }


class PhaseManager:
    """分阶段训练管理器"""

    def __init__(self, config):
        self.config = config
        self.phases = config['TRAINING_PHASES']
        self.current_phase = 0
        self.phase_start_step = 0
        self.phase_completed = False

        # 创建阶段日志文件
        self.phase_log_file = os.path.join(config['LOG_DIR'], 'phase_log.json')
        self._log_phase_start()

    def check_phase_transition(self, step, eval_metrics):
        """检查是否应该转换阶段"""
        if self.current_phase >= len(self.phases):
            return False

        current_phase_info = self.phases[self.current_phase]
        phase_duration = step - self.phase_start_step

        # 检查最短时间要求
        if phase_duration < current_phase_info['min_steps']:
            return False

        # 检查最长时间限制
        if phase_duration >= current_phase_info['max_steps']:
            return True

        # 检查成功条件
        return self._check_success_criteria(eval_metrics, current_phase_info['success_criteria'])

    def _check_success_criteria(self, eval_metrics, criteria):
        """检查阶段成功条件"""
        if not eval_metrics:
            return False

        for criterion, threshold in criteria.items():
            if criterion == 'min_episode_length':
                if eval_metrics.get('eval_mean_length', 0) < threshold:
                    return False
            elif criterion == 'max_localization_error':
                if eval_metrics.get('mean_localization_error', float('inf')) > threshold:
                    return False
            elif criterion == 'min_coordination_score':
                if eval_metrics.get('coordination_score', 0) < threshold:
                    return False
            elif criterion == 'convergence_rate':
                if eval_metrics.get('convergence_rate', 0) < threshold:
                    return False

        return True

    def advance_phase(self, step, trainer):
        """进入下一阶段"""
        # 保存当前阶段模型
        current_phase_info = self.phases[self.current_phase]
        model_path = os.path.join(
            self.config['LOG_DIR'],
            f"model_phase_{self.current_phase}_{current_phase_info['name']}.pt"
        )
        trainer.save_model(model_path)

        # 记录阶段完成
        self._log_phase_completion(step)

        # 进入下一阶段
        self.current_phase += 1
        self.phase_start_step = step

        if self.current_phase < len(self.phases):
            # 应用新阶段配置
            self._apply_phase_config(trainer)
            self._log_phase_start()
            return True
        else:
            print("All phases completed!")
            return False

    def _apply_phase_config(self, trainer):
        """应用阶段特定配置"""
        if self.current_phase >= len(self.phases):
            return

        phase_info = self.phases[self.current_phase]
        config_overrides = phase_info['config_overrides']

        print(f"\n{'=' * 60}")
        print(f"ENTERING PHASE {self.current_phase + 1}: {phase_info['name']}")
        print(f"Description: {phase_info['description']}")
        print(f"Expected duration: {config_overrides.get('max_steps', 'unlimited')} steps")
        print(f"{'=' * 60}")

        # 更新学习率
        if 'ACTOR_LR' in config_overrides:
            new_lr = config_overrides['ACTOR_LR']
            for param_group in trainer.algorithm.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"✓ Actor learning rate: {new_lr}")

        # 更新其他算法参数
        if 'ENTROPY_COEF' in config_overrides:
            trainer.algorithm.entropy_coef = config_overrides['ENTROPY_COEF']
            print(f"✓ Entropy coefficient: {config_overrides['ENTROPY_COEF']}")

        if 'EPS_CLIP' in config_overrides:
            trainer.algorithm.eps_clip = config_overrides['EPS_CLIP']
            print(f"✓ PPO clip parameter: {config_overrides['EPS_CLIP']}")

        # 更新缓冲区大小（如果需要）
        if 'ROLLOUT_LENGTH' in config_overrides:
            new_rollout_length = config_overrides['ROLLOUT_LENGTH']
            print(f"✓ Rollout length: {new_rollout_length}")
            print("  (Note: Rollout length change requires buffer recreation)")

        print(f"{'=' * 60}\n")

    def _log_phase_start(self):
        """记录阶段开始"""
        if self.current_phase < len(self.phases):
            phase_info = self.phases[self.current_phase]
            log_entry = {
                'event': 'phase_start',
                'phase': self.current_phase,
                'phase_name': phase_info['name'],
                'start_step': self.phase_start_step,
                'timestamp': datetime.now().isoformat(),
                'config': phase_info['config_overrides']
            }
            self._append_to_log(log_entry)

    def _log_phase_completion(self, step):
        """记录阶段完成"""
        phase_info = self.phases[self.current_phase]
        duration = step - self.phase_start_step
        log_entry = {
            'event': 'phase_completion',
            'phase': self.current_phase,
            'phase_name': phase_info['name'],
            'completion_step': step,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        self._append_to_log(log_entry)

    def _append_to_log(self, log_entry):
        """追加日志条目"""
        if os.path.exists(self.phase_log_file):
            with open(self.phase_log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)

        with open(self.phase_log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def get_current_phase_info(self):
        """获取当前阶段信息"""
        if self.current_phase < len(self.phases):
            return self.phases[self.current_phase]
        return None

    def get_progress_info(self, current_step):
        """获取进度信息"""
        if self.current_phase >= len(self.phases):
            return "Training completed"

        phase_info = self.phases[self.current_phase]
        phase_duration = current_step - self.phase_start_step
        progress = phase_duration / phase_info['max_steps'] * 100

        return {
            'current_phase': self.current_phase + 1,
            'total_phases': len(self.phases),
            'phase_name': phase_info['name'],
            'phase_progress': f"{progress:.1f}%",
            'phase_duration': phase_duration,
            'phase_max_steps': phase_info['max_steps']
        }