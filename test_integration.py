#!/usr/bin/env python3

"""
Test script to verify the localization environment works with PyMARL.
"""

import sys
import os
import numpy as np

# Add pymarl source to Python path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pymarl_src'))

from localization_env import LocalizationEnv

def test_localization_env():
    """Test the localization environment."""
    print("Testing LocalizationEnv with PyMARL interface...")
    
    # Create environment
    env = LocalizationEnv()
    
    # Test environment info
    env_info = env.get_env_info()
    print(f"Environment info: {env_info}")
    
    # Test reset
    obs, state = env.reset()
    print(f"Initial obs shape: {[o.shape for o in obs]}")
    print(f"Initial state shape: {state.shape}")
    
    # Test a few steps
    for step in range(5):
        # Random actions
        actions = [np.random.randint(0, env.get_total_actions()) for _ in range(env.n_agents)]
        
        # Execute step
        reward, terminated, info = env.step(actions)
        
        # Get observations and state
        obs = env.get_obs()
        state = env.get_state()
        
        print(f"Step {step + 1}:")
        print(f"  Actions: {actions}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Terminated: {terminated}")
        print(f"  Info keys: {list(info.keys())}")
        
        if terminated:
            print("Episode terminated!")
            break
    
    # Test stats
    stats = env.get_stats()
    print(f"Final stats: {stats}")
    
    print("✅ LocalizationEnv test completed successfully!")

def test_environment_registration():
    """Test environment registration in PyMARL."""
    print("\nTesting environment registration...")
    
    from pymarl_src.envs import REGISTRY
    
    print(f"Available environments: {list(REGISTRY.keys())}")
    
    if "localization" in REGISTRY:
        print("✅ Localization environment successfully registered!")
        
        # Test creating environment through registry
        env_fn = REGISTRY["localization"]
        env = env_fn()
        print(f"Environment created through registry: {type(env)}")
        
        # Quick test
        obs, state = env.reset()
        print(f"Registry env test - obs: {len(obs)}, state: {state.shape}")
        
    else:
        print("❌ Localization environment not found in registry!")

if __name__ == "__main__":
    test_localization_env()
    test_environment_registration()