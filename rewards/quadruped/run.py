"""module for retrieving ru  reward function from dm_control suite quadruped"""

from custom_dmc_tasks.quadruped import run

reward_function = run()._task.get_reward  # pylint: disable=protected-access
