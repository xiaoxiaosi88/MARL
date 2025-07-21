#!/usr/bin/env python3

"""
Demo script to run a short training session with the localization environment.
This demonstrates PyMARL training integration.
"""

import sys
import os
import numpy as np

# Add pymarl source to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pymarl_src'))

def demo_training():
    """Run a short demo training session."""
    print("=== PyMARL Localization Environment Training Demo ===")
    
    # Test basic environment functionality
    from localization_env import LocalizationEnv
    
    env = LocalizationEnv()
    print(f"Environment loaded with {env.n_agents} agents")
    print(f"Observation size: {env.get_obs_size()}")
    print(f"State size: {env.get_state_size()}")
    print(f"Action space: {env.get_total_actions()}")
    
    # Run a few episodes to demonstrate functionality
    print("\n--- Running demonstration episodes ---")
    
    for episode in range(3):
        obs, state = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while step_count < 20:  # Limit steps for demo
            # Random actions for demo
            actions = [np.random.randint(0, env.get_total_actions()) for _ in range(env.n_agents)]
            
            reward, terminated, info = env.step(actions)
            episode_reward += reward
            step_count += 1
            
            if step_count % 5 == 0:
                stats = env.get_stats()
                print(f"  Step {step_count}: Reward={reward:.2f}, Mean Error={stats.get('mean_localization_error', 0):.2f}")
            
            if terminated:
                print(f"  Episode terminated at step {step_count}")
                break
        
        print(f"  Total episode reward: {episode_reward:.2f}")
    
    print("\n✅ Demo completed successfully!")
    print("\nTo run full training with PyMARL algorithms, use:")
    print("  python train_localization.py --config=qmix --env-config=localization")
    print("  python train_localization.py --config=vdn --env-config=localization")
    print("  python train_localization.py --config=iql --env-config=localization")

def show_config_info():
    """Show available configuration information."""
    print("\n=== Available PyMARL Configurations ===")
    
    # Show available algorithm configs
    config_dir = os.path.join(os.path.dirname(__file__), 'pymarl_src', 'config', 'algs')
    if os.path.exists(config_dir):
        alg_configs = [f.replace('.yaml', '') for f in os.listdir(config_dir) if f.endswith('.yaml')]
        print(f"Available algorithms: {', '.join(alg_configs)}")
    
    # Show environment config
    env_config_path = os.path.join(os.path.dirname(__file__), 'pymarl_src', 'config', 'envs', 'localization.yaml')
    if os.path.exists(env_config_path):
        print(f"Environment config: {env_config_path}")
        with open(env_config_path, 'r') as f:
            print("Environment configuration:")
            print(f.read())

if __name__ == "__main__":
    demo_training()
    show_config_info()