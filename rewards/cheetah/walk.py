"""
module for retrieving reach_bottom_left reward function
from dm_control suite point_mass_maze
"""

from custom_dmc_tasks.cheetah import walk

reward_function = walk()._task.get_reward  # pylint: disable=protected-access
