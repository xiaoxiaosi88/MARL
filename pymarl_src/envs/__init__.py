from functools import partial
import sys
import os

try:
    from smac.env import MultiAgentEnv, StarCraft2Env
    SMAC_AVAILABLE = True
except ImportError:
    print("SMAC not available, only localization environment will be available")
    SMAC_AVAILABLE = False
    from .multiagentenv import MultiAgentEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

def localization_env_fn(**kwargs):
    """Create localization environment avoiding circular imports."""
    import sys
    import os
    
    # Get the parent directory to import from
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from localization_env import LocalizationEnv
    return LocalizationEnv(**kwargs)

REGISTRY = {}

# Register localization environment
REGISTRY["localization"] = localization_env_fn

# Register SMAC if available
if SMAC_AVAILABLE:
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
