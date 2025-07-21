"""
PyMARL wrapper for the localization environment.
This adapter converts the PettingZoo-style localization environment to work with PyMARL.
"""

import numpy as np
from pymarl_src.envs.multiagentenv import MultiAgentEnv
from localization import MyNewMultiAgentEnv


class LocalizationEnv(MultiAgentEnv):
    """PyMARL wrapper for the localization environment."""
    
    def __init__(self, **kwargs):
        # Remove PyMARL-specific arguments that aren't needed for the base environment
        pymarl_args = ['seed', 'map_name']
        env_kwargs = {k: v for k, v in kwargs.items() if k not in pymarl_args}
        
        # Initialize the base localization environment
        self.env = MyNewMultiAgentEnv(**env_kwargs)
        
        # PyMARL expects specific attributes
        self.n_agents = self.env.num_agents
        self.episode_limit = self.env.max_episode_steps
        
        # Cache environment info
        self._obs_size = self.env.obs_dim
        self._state_size = self.env.obs_dim * self.n_agents  # Global state is concatenated observations
        self._n_actions = 9  # Discrete actions 0-8
        
        # Track current step data
        self._obs = None
        self._state = None
        self._terminated = False
        self._truncated = False
        self._info = {}
        
    def reset(self):
        """Reset the environment and return initial observations and state."""
        obs_dict, info_dict = self.env.reset()
        
        # Convert PettingZoo format to PyMARL format
        self._obs = [obs_dict[f"agent_{i}"] for i in range(self.n_agents)]
        self._state = self.env.state()
        self._terminated = False
        self._truncated = False
        self._info = info_dict
        
        return self.get_obs(), self.get_state()
    
    def step(self, actions):
        """Execute actions and return results in PyMARL format."""
        # Convert actions list to PettingZoo action dict
        action_dict = {f"agent_{i}": actions[i] for i in range(self.n_agents)}
        
        # Execute step in the environment
        obs_dict, rewards_dict, terminated_dict, truncated_dict, info_dict = self.env.step(action_dict)
        
        # Convert to PyMARL format
        self._obs = [obs_dict[f"agent_{i}"] for i in range(self.n_agents)]
        self._state = self.env.state()
        
        # PyMARL expects single reward, terminated, and info
        reward = sum(rewards_dict.values()) / self.n_agents  # Average reward
        terminated = any(terminated_dict.values())
        self._terminated = terminated
        self._truncated = any(truncated_dict.values())
        
        # Combine info from all agents
        self._info = {}
        for i, agent_key in enumerate([f"agent_{i}" for i in range(self.n_agents)]):
            if agent_key in info_dict:
                for key, value in info_dict[agent_key].items():
                    if key not in self._info:
                        self._info[key] = []
                    self._info[key].append(value)
        
        return reward, terminated, self._info
    
    def get_obs(self):
        """Return all agent observations in a list."""
        return self._obs
    
    def get_obs_agent(self, agent_id):
        """Return observation for specific agent."""
        return self._obs[agent_id] if self._obs is not None else None
    
    def get_obs_size(self):
        """Return the size of the observation."""
        return self._obs_size
    
    def get_state(self):
        """Return the global state."""
        return self._state
    
    def get_state_size(self):
        """Return the size of the state."""
        return self._state_size
    
    def get_avail_actions(self):
        """Return available actions for all agents."""
        # All actions are always available in the localization environment
        return [[1] * self._n_actions for _ in range(self.n_agents)]
    
    def get_avail_agent_actions(self, agent_id):
        """Return available actions for specific agent."""
        # All actions are always available
        return [1] * self._n_actions
    
    def get_total_actions(self):
        """Return total number of actions."""
        return self._n_actions
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        return self.env.close()
    
    def seed(self, seed=None):
        """Set the random seed."""
        self.env.reset(seed=seed)
        return [seed]
    
    def save_replay(self):
        """Save replay (not implemented for localization environment)."""
        pass
    
    def get_env_info(self):
        """Return environment information for PyMARL."""
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(), 
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
        return env_info
    
    def get_stats(self):
        """Return environment statistics."""
        if not self._info:
            return {}
        
        stats = {}
        # Calculate statistics from the last step info
        if 'localization_error' in self._info:
            errors = self._info['localization_error']
            stats['mean_localization_error'] = np.mean(errors)
            stats['max_localization_error'] = np.max(errors) 
            stats['min_localization_error'] = np.min(errors)
        
        if 'global_loss' in self._info and self._info['global_loss']:
            stats['global_loss'] = self._info['global_loss'][0]  # Same for all agents
            
        if 'boundary_violation' in self._info:
            stats['boundary_violations'] = sum(self._info['boundary_violation'])
        
        return stats