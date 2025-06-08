"""Custom DeepMind Control Suite tasks."""

from custom_dmc_tasks import cheetah
from custom_dmc_tasks import walker
from custom_dmc_tasks import hopper
from custom_dmc_tasks import quadruped
from custom_dmc_tasks import jaco
from custom_dmc_tasks import point_mass_maze
from dm_control.suite.wrappers import pixels


def make(
    domain,
    task,
    obs_type: str,
    frames: int,
    task_kwargs=None,
    environment_kwargs=None,
    visualize_reward=False,
):
    assert obs_type in ["states", "pixels", "perfect_features"]

    if domain in ("cheetah", "cheetah_occluded"):
        env = cheetah.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    elif domain in ("walker", "walker_occluded"):
        env = walker.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    elif domain in (
        "point_mass_maze",
        "point_mass_maze_simplified",
        "point_mass_maze_simplified_dense",
        "point_mass_maze_occluded",
        "point_mass_maze_hard",
        "point_mass_maze_dense",
        "point_mass_maze_obstacle_top",
        "point_mass_maze_obstacle_left",
        "point_mass_maze_obstacle_right",
        "point_mass_maze_obstacle_box",
    ):
        if domain == "point_mass_maze_obstacle_top":
            obstacle = "top"
        elif domain == "point_mass_maze_obstacle_left":
            obstacle = "left"
        elif domain == "point_mass_maze_obstacle_right":
            obstacle = "right"
        elif domain == "point_mass_maze_obstacle_box":
            obstacle = "box"
        else:
            obstacle = None
        env = point_mass_maze.make(  # pylint: disable=E1123
            task,
            obstacle=obstacle,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    elif domain == "hopper":
        env = hopper.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    elif domain in ("quadruped", "quadruped_occluded"):
        env = quadruped.make(
            task,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            visualize_reward=visualize_reward,
        )
    elif domain in ("jaco", "jaco_occluded"):
        env = jaco.make(
            task,
            obs_type="perfect_features",
            seed=42,
        )
    else:
        raise f"{domain} not found"

    if obs_type == "pixels":
        from dmc import FrameStackWrapper  # pylint: disable=import-outside-toplevel

        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)  # pylint: disable=R1735
        render_kwargs = dict(  # pylint: disable=R1735
            height=64, width=64, camera_id=camera_id
        )
        env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs)
        env = FrameStackWrapper(env, frames, pixels_key="pixels")

    return env


def make_jaco(task, obs_type, seed):
    return jaco.make(task, obs_type, seed)
