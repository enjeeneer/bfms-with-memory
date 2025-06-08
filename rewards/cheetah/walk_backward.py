"""
module for retrieving walk_backward reward function
from dm_control suite cheetah
"""

from custom_dmc_tasks.cheetah import walk_backward

reward_function = walk_backward()._task.get_reward  # pylint: disable=protected-access
