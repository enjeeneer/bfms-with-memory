"""
module for retrieving reach_bottom_left reward function
from dm_control suite point_mass_maze
"""

from custom_dmc_tasks.cheetah import run

reward_function = run()._task.get_reward  # pylint: disable=protected-access
