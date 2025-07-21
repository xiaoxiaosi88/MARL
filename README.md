# MARL Localization Environment with PyMARL Integration

This repository contains a multi-agent localization environment integrated with the PyMARL (Python Multi-Agent Reinforcement Learning) framework.

## Overview

The localization environment (`localization.py`) is a multi-agent reinforcement learning environment where agents collaborate to localize sensor positions using distance measurements from anchors and neighboring sensors. The environment has been successfully integrated with PyMARL to enable training with state-of-the-art multi-agent reinforcement learning algorithms.

## Environment Description

### Localization Problem
- **8 mobile sensors** need to estimate their positions
- **4 fixed anchors** provide distance measurements
- Sensors can communicate with neighbors within communication range
- Goal: Minimize localization error through coordinated movement

### Key Features
- **State Space**: Each agent observes its position, neighbor positions, and distance measurements
- **Action Space**: 9 discrete actions (stay, up, down, left, right, diagonals)
- **Reward**: Negative global loss function (agents collaborate to minimize total error)
- **Episode Termination**: Convergence to target accuracy or maximum steps reached

## Installation

### Dependencies
```bash
pip install numpy pettingzoo gymnasium networkx torch sacred pymongo tensorboard-logger scipy pillow
```

### Quick Setup
1. Clone the repository
2. Install dependencies
3. Run demo or training scripts

## Usage

### Quick Demo
```bash
python demo_training.py
```

### Training with PyMARL

#### Independent Q-Learning (IQL)
```bash
cd pymarl_src
python main.py --config=iql --env-config=localization
```

#### QMIX Algorithm
```bash
cd pymarl_src  
python main.py --config=qmix --env-config=localization
```

#### VDN (Value Decomposition Networks)
```bash
cd pymarl_src
python main.py --config=vdn --env-config=localization
```

#### COMA (Counterfactual Multi-Agent Policy Gradients)
```bash
cd pymarl_src
python main.py --config=coma --env-config=localization
```

### Custom Training Parameters
```bash
cd pymarl_src
python main.py --config=qmix --env-config=localization with t_max=50000 test_interval=2000
```

### Available Algorithms
- **IQL**: Independent Q-Learning
- **QMIX**: Monotonic Value Function Factorisation
- **VDN**: Value-Decomposition Networks
- **COMA**: Counterfactual Multi-Agent Policy Gradients
- **QTRAN**: Learning to Factorize with Transformation

## Environment Configuration

The environment can be customized through `pymarl_src/config/envs/localization.yaml`:

```yaml
env_args:
  communication_range: 250          # Communication range between sensors
  sensor_noise_std: 5.0            # Noise in sensor-to-sensor measurements
  anchor_noise_std: 3.0            # Noise in anchor-to-sensor measurements  
  max_episode_steps: 100           # Maximum steps per episode
  action_magnitude: 1.0            # Step size for movements
  boundary_penalty: -100.0         # Penalty for boundary violations
  convergence_threshold: 2         # Target localization accuracy
```

## File Structure

```
.
├── localization.py              # Original PettingZoo environment
├── localization_env.py          # PyMARL adapter wrapper
├── train_localization.py        # Training script 
├── demo_training.py             # Demo script
├── test_integration.py          # Integration test
└── pymarl_src/                  # PyMARL framework
    ├── main.py                  # Main training entry point
    ├── config/
    │   ├── algs/               # Algorithm configurations
    │   └── envs/               # Environment configurations
    ├── envs/                   # Environment interfaces
    ├── learners/               # Learning algorithms
    ├── modules/                # Neural network modules
    └── runners/                # Training runners
```

## Integration Details

### PyMARL Adapter
The `LocalizationEnv` class in `localization_env.py` wraps the original PettingZoo environment to work with PyMARL's interface:

- Converts PettingZoo observation dictionaries to PyMARL observation lists
- Handles action space conversion (discrete actions)
- Provides global state for centralized training methods
- Computes environment statistics for logging

### Key Differences from PettingZoo
- **Observations**: Lists indexed by agent ID instead of dictionaries
- **Actions**: Single list instead of dictionary
- **Rewards**: Single scalar (averaged) instead of per-agent rewards
- **Global State**: Available for centralized training algorithms

## Testing

### Integration Test
```bash
python test_integration.py
```

### Environment Demo
```bash
python demo_training.py
```

## Results and Logs

Training results are saved in `results/sacred/` with:
- Configuration files
- Training metrics
- Console output logs
- Model checkpoints

## Monitoring Training

Logs include:
- Episode rewards
- Localization errors
- Boundary violations
- Training loss
- Test performance

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Warnings**: Normal for CPU-only training
3. **Sacred Logs**: Debug output is normal for experiment tracking

### Environment Verification
```python
from localization_env import LocalizationEnv

env = LocalizationEnv()
obs, state = env.reset()
print(f"Environment working: {len(obs)} agents, state shape {state.shape}")
```

## Performance Notes

- Training typically takes 10K-100K timesteps depending on algorithm
- IQL converges fastest but may get stuck in local optima
- QMIX shows better coordination but requires more training time
- Adjust `t_max`, `test_interval`, and learning rates based on your needs

## Citation

If you use this environment, please cite the original PyMARL framework:

```bibtex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```