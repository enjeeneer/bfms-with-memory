# pylint: disable=line-too-long, unused-argument

"""Module that creates workspaces for training/evaling various agents."""
import wandb
import torch
import gymnasium
import shutil
from os import makedirs
from loguru import logger
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple, Any
from scipy import stats
from dataclasses import asdict
from utils import (
    BASE_DIR,
    upload_to_gcs_bucket,
    GOAL_BASED_EXORL_DOMAINS,
    VideoRecorder,
)
import matplotlib.pyplot as plt
from popgym.core.env import POPGymEnv

from rewards import RewardFunctionConstructor
from agents.base import AbstractWorkspace
from agents.fb.agent import FB

from agents.rsf.agent import RSF

from agents.base import (
    MemoryEfficientOfflineReplayBuffer,
)
from agents.rfb.agent import RFB
from agents.rnd.agent import RND

class ExorlWorkspace:
    """
    Trains/evals/rollouts an agent on Exorl.
    """

    def __init__(
        self,
        reward_constructors: Dict[Tuple[float, float], RewardFunctionConstructor],
        learning_steps: int,
        model_dir: Path,
        eval_frequency: int,
        eval_rollouts: int,
        wandb_logging: bool,
        device: torch.device,
        eval_multipliers: List[float],
        dynamics_occlusion: callable,
        reward_occlusion: callable,
        goal_frames: int,
        wandb_project: str,
        wandb_entity: str,
        z_inference_steps: Optional[int] = None,  # FB only
        train_std: Optional[float] = None,  # FB only
        eval_std: Optional[float] = None,  # FB only
        save_model: bool = False,
        save_multiplier: float = None,  # save w.r.t. perf on eval multiplier
    ):

        self.reward_constructors = reward_constructors
        self.eval_frequency = eval_frequency  # how frequently to eval
        self.eval_rollouts = eval_rollouts  # how many rollouts per eval step
        self.model_dir = model_dir
        self.learning_steps = learning_steps
        self.z_inference_steps = z_inference_steps
        self.train_std = train_std
        self.eval_std = eval_std
        self.dynamics_occlusion = dynamics_occlusion
        self.reward_occlusion = reward_occlusion
        self.goal_frames = goal_frames
        self.goals_z = None
        self.rewards_z = None
        self.wandb_logging = wandb_logging
        self.eval_multipliers = eval_multipliers
        self.domain_name = reward_constructors[
            (eval_multipliers[0], eval_multipliers[0])
        ].domain_name
        self.device = device
        self.save_model = save_model
        self.save_multiplier = save_multiplier
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

        # get task names from reward constructor
        for _, rc in reward_constructors.items():
            self.tasks = rc.task_names
            break

    def train(
        self,
        agent: Union[FB, RSF, RFB],
        agent_config: Dict,
        replay_buffer: MemoryEfficientOfflineReplayBuffer,
    ) -> None:
        """
        Trains an agent provided a replay buffer.
        """
        if self.wandb_logging:
            run = wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                config=agent_config,
                tags=[agent.name],
                reinit=True,
            )
            model_path = self.model_dir / run.name
            makedirs(str(model_path))
            run_name = run.name

        else:
            date = datetime.today().strftime("Y-%m-%d-%H-%M-%S")
            run_name = f"local-run-{date}"
            model_path = self.model_dir / run_name
            makedirs(str(model_path))

        logger.info(f"Training {agent.name}.")
        best_mean_task_reward = -np.inf
        best_model_path = None

        # sample set transitions for z inference
        if isinstance(agent, (FB, SF, GCIQL)):
            if "point_mass_maze" in self.domain_name:
                self.goal_states = {}
                for task, goal_state in point_mass_maze_goals.items():

                    # we don't apply reward occlusion to goal-reaching
                    # task definition
                    self.goal_states[task] = torch.tensor(
                        np.concatenate([goal_state] * self.goal_frames),
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)

        if isinstance(agent, (RFB, RSF)):
            if "point_mass_maze" in self.domain_name:
                self.goal_states = {}
                for task, goal_state in point_mass_maze_goals.items():
                    # we don't apply reward occlusion to goal-reaching
                    # task definition
                    self.goal_states[task] = torch.tile(
                        torch.tensor(
                            goal_state,
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        (1, agent.backward_history_length, 1),
                    )

            # if agent.observation_type == "states":
            #     (
            #         self.goals_z,
            #         self.rewards_z,
            #         self.positions_z,
            #     ) = replay_buffer.sample_task_inference_transitions(
            #         inference_steps=self.z_inference_steps,
            #     )
        # for pixels we've already loaded the goals/rewards
        if agent.observation_type == "pixels":
            self.goals_z = replay_buffer._storage["goals_z"]  # pylint: disable=W0212
            self.rewards_z = replay_buffer._storage[  # pylint: disable=W0212
                "rewards_z"
            ]

        for i in tqdm(range(self.learning_steps + 1)):

            batch = replay_buffer.sample(
                batch_size=agent.batch_size,
                return_rewards=isinstance(agent, TD3GRU),
            )
            train_metrics = agent.update(batch=batch, step=i)

            eval_metrics = {}

            if i % self.eval_frequency == 0:
                eval_metrics = self.eval(
                    agent=agent,
                    best_mean_task_reward=best_mean_task_reward,
                    run_name=run_name,
                    replay_buffer=replay_buffer,
                )

                if (
                    self.save_model
                    and eval_metrics[
                        f"all_tasks/mass={self.save_multiplier}x; "
                        f"damping={self.save_multiplier}x"
                    ]
                    > best_mean_task_reward
                ):
                    new_best_mean_task_reward = eval_metrics[
                        f"all_tasks/mass={self.save_multiplier}x; "
                        f"damping={self.save_multiplier}x"
                    ]
                    logger.info(
                        f"New max IQM task reward: {best_mean_task_reward:.3f} -> "
                        f"{new_best_mean_task_reward:.3f}."
                        f" Saving model."
                    )

                    # delete current best model
                    if best_model_path is not None:
                        best_model_path.unlink(missing_ok=True)

                    agent._name = i  # pylint: disable=protected-access
                    # save locally
                    best_model_path = agent.save(model_path)

                    best_mean_task_reward = new_best_mean_task_reward

                agent.train()

            metrics = {**train_metrics, **eval_metrics}

            if self.wandb_logging:
                run.log(metrics)

        if self.wandb_logging:
            if self.save_model:
                # upload model to wandb at end of training
                run.save(best_model_path.as_posix(), base_path=model_path.as_posix())
            run.finish()

        # delete local models
        shutil.rmtree(model_path)

    def eval(
        self,
        agent: Union[CQL, SAC, FB, CFB, CalFB, SF, RFB, GCIQL, TD3GRU],
        best_mean_task_reward: float,
        run_name: str,
        replay_buffer: MemoryEfficientOfflineReplayBuffer,
    ) -> Dict[str, float]:
        """
        Performs eval rollouts.
        Args:
            agent: agent to evaluate
        Returns:
            metrics: dict of metrics
        """

        if isinstance(agent, (FB, SF, RFB, RSF, GCIQL)):
            zs = {}

            # fixed inference dataset
            for multiplier in self.eval_multipliers:
                env_variant_zs = {}

                # goals = self.goals_z[(multiplier, multiplier)].to(self.device)
                # rewards_dict = self.rewards_z[(multiplier, multiplier)]
                # positions = self.positions_z[(multiplier, multiplier)]

                if self.domain_name in GOAL_BASED_EXORL_DOMAINS:
                    if isinstance(agent, (FB, RFB, GCIQL)):
                        env_variant_zs = agent.infer_z(
                            replay_buffer=replay_buffer,
                            goal_state_dict=self.goal_states,
                            multiplier=multiplier,
                        )
                else:
                    env_variant_zs = agent.infer_z(
                        replay_buffer=replay_buffer,
                        multiplier=multiplier,
                    )

                zs[(multiplier, multiplier)] = env_variant_zs

                agent.std_dev_schedule = self.eval_std

        logger.info("Performing eval rollouts.")
        metrics = {}
        agent.eval()

        # loop over environment variants
        for multiplier in self.eval_multipliers:
            env_variant_eval_rewards = {}
            env = self.reward_constructors[(multiplier, multiplier)].env
            reward_functions = self.reward_constructors[
                (multiplier, multiplier)
            ].reward_functions

            if isinstance(agent, (FB, RFB, GCIQL)):
                env_zs = zs[(multiplier, multiplier)]
            elif (
                isinstance(agent, (SF, RSF))
                and self.domain_name not in GOAL_BASED_EXORL_DOMAINS
            ):
                env_zs = zs[(multiplier, multiplier)]

            for _ in tqdm(
                range(self.eval_rollouts),
                desc=f"eval rollouts for {multiplier}x"
                f" mass & {multiplier}x damping",
            ):
                for task in self.tasks:
                    task_rewards = 0.0

                    timestep = env.reset()
                    step = 0
                    if isinstance(agent, (TD3GRU, RFB, RSF)):
                        if not agent.inference_memory:
                            if isinstance(agent, (RFB, RSF)):
                                if agent.recurrent_actor:
                                    if agent.memory_type == "transformer":
                                        if agent.transformer_attention == "flash":
                                            (
                                                prev_obs_action_internal_state,
                                                prev_obs_z_internal_state,
                                            ) = agent.actor.init_internal_state()
                                        else:
                                            prev_obs_action_internal_state = None
                                            prev_obs_z_internal_state = None
                                    else:
                                        (
                                            prev_obs_action_internal_state,
                                            prev_obs_z_internal_state,
                                        ) = agent.actor.init_internal_state()
                                else:
                                    prev_obs_action_internal_state = None
                                    prev_obs_z_internal_state = None
                            else:
                                (
                                    prev_obs_action_internal_state,
                                    prev_obs_z_internal_state,
                                ) = agent.actor.init_internal_state()
                        else:
                            prev_obs_action_internal_state = None
                            prev_obs_z_internal_state = None

                    while not timestep.last():
                        # different naming for pixels vs states
                        try:
                            observation = self.dynamics_occlusion(
                                timestep.observation["observations"]
                            )
                        except:  # pylint: disable=bare-except
                            observation = self.dynamics_occlusion(timestep.observation)

                        if isinstance(agent, (FB, GCIQL)):
                            action, _ = agent.act(
                                observation,
                                task=env_zs[task],
                                step=step,
                                sample=False,
                            )
                        elif isinstance(agent, (RSF, SF)):
                            if self.domain_name not in GOAL_BASED_EXORL_DOMAINS:
                                action, _ = agent.act(
                                    observation,
                                    task=env_zs[task],
                                    step=step,
                                    sample=False,
                                )
                            # calculate z at every step
                            else:
                                z = agent.infer_z_from_goal(
                                    observation=self.reward_occlusion(
                                        timestep.observation["observations"]
                                    ),
                                    goal_state=self.goal_states[task],
                                    step=0,
                                )
                                action, _ = agent.act(
                                    observation,
                                    task=z,
                                    step=step,
                                    sample=False,
                                )

                        elif isinstance(agent, TD3GRU):
                            action, prev_obs_action_internal_state = agent.act(
                                observation,
                                step=step,
                                sample=False,
                                previous_internal_state=prev_obs_action_internal_state,
                            )
                        elif isinstance(agent, RFB):
                            (
                                action,
                                prev_obs_action_internal_state,
                                prev_obs_z_internal_state,
                            ) = agent.act(
                                observation=observation,
                                task=env_zs[task],
                                step=step,
                                sample=False,
                                previous_obs_action_internal_state=prev_obs_action_internal_state,  # pylint: disable=CO301
                                previous_obs_z_internal_state=prev_obs_z_internal_state,
                            )

                        # non zero-shot RL methods
                        else:
                            action = agent.act(
                                observation,
                                sample=False,
                                step=None,
                            )
                        timestep = env.step(action)
                        task_rewards += reward_functions[task](env.physics)
                        step += 1

                    if task not in env_variant_eval_rewards:
                        env_variant_eval_rewards[task] = []

                    # normalise jaco rewards
                    task_rewards = (
                        task_rewards * 4 if self.domain_name == "jaco" else task_rewards
                    )
                    env_variant_eval_rewards[task].append(task_rewards)

            # average over rollouts for each environment variant and log
            mean_env_variant_task_performance = 0.0
            for task, rewards in env_variant_eval_rewards.items():
                mean_task_reward = stats.trim_mean(rewards, 0.25)
                metrics[
                    f"{task}/mass={multiplier}x; " f"damping={multiplier}x"
                ] = mean_task_reward
                mean_env_variant_task_performance += mean_task_reward

            metrics[
                f"all_tasks/mass={multiplier}x; " f"damping={multiplier}x"
            ] = mean_env_variant_task_performance / len(self.tasks)

        # save video if new rollout reward > best rollout reward
        save_string = "all_tasks/mass=1.0x; damping=1.0x"
        if metrics[save_string] > best_mean_task_reward:
            logger.info(
                f"New max IQM task reward: {best_mean_task_reward:.3f} "
                f"-> "
                f"{metrics[save_string]}"
            )
            for task in self.tasks:
                logger.info(f"Recording video for {task}")
                self.record_rollout(
                    agent=agent,
                    z_goal_state=env_zs[task]
                    if isinstance(agent, (FB, RFB, SF, RSF))
                    else None,
                    task=task,
                    run_name=run_name,
                )

        if isinstance(agent, (FB, SF, RFB)):
            agent.std_dev_schedule = self.train_std

        return metrics

    def record_rollout(
        self,
        agent,
        z_goal_state,
        task: str,
        run_name: str,
    ):
        """
        Records video of rollout.
        """
        env = self.reward_constructors[(1.0, 1.0)].env

        video_recorder = VideoRecorder(root_dir=BASE_DIR)
        video_recorder.enabled = True
        video_recorder.init(env=env)
        timestep = env.reset()
        step = 0
        if isinstance(agent, (TD3GRU, RFB, RSF)):
            if not agent.inference_memory:
                if isinstance(agent, (RFB, RSF)):
                    if agent.recurrent_actor:
                        if agent.memory_type == "transformer":
                            if agent.transformer_attention == "flash":
                                (
                                    prev_obs_action_internal_state,
                                    prev_obs_z_internal_state,
                                ) = agent.actor.init_internal_state()
                            else:
                                prev_obs_action_internal_state = None
                                prev_obs_z_internal_state = None
                        else:
                            (
                                prev_obs_action_internal_state,
                                prev_obs_z_internal_state,
                            ) = agent.actor.init_internal_state()
                    else:
                        prev_obs_action_internal_state = None
                        prev_obs_z_internal_state = None
                else:
                    prev_obs_action_internal_state = agent.actor.init_internal_state()
            else:
                prev_obs_action_internal_state = None
                prev_obs_z_internal_state = None

        while not timestep.last():
            video_recorder.record(env)

            # different naming for pixels vs states
            try:
                observation = self.dynamics_occlusion(
                    timestep.observation["observations"]
                )
            except:  # pylint: disable=bare-except
                observation = self.dynamics_occlusion(timestep.observation)

            if isinstance(agent, (FB, GCIQL)):
                action, _ = agent.act(
                    observation,
                    task=z_goal_state,
                    step=step,
                    sample=False,
                )
            elif isinstance(agent, (SF, RSF)):
                if self.domain_name not in GOAL_BASED_EXORL_DOMAINS:
                    action, _ = agent.act(
                        observation,
                        task=z_goal_state,
                        step=step,
                        sample=False,
                    )
                # calculate z at every step
                else:
                    z = agent.infer_z_from_goal(
                        observation=self.reward_occlusion(
                            timestep.observation["observations"]
                        ),
                        goal_state=self.goal_states[task],
                        step=0,
                    )
                    action, _ = agent.act(
                        observation,
                        task=z,
                        step=step,
                        sample=False,
                    )

            elif isinstance(agent, TD3GRU):
                action, prev_obs_action_internal_state = agent.act(
                    observation,
                    step=step,
                    sample=False,
                    previous_internal_state=prev_obs_action_internal_state,
                )
            elif isinstance(agent, RFB):

                (
                    action,
                    prev_obs_action_internal_state,
                    prev_obs_z_internal_state,
                ) = agent.act(
                    observation=observation,
                    task=z_goal_state,
                    step=step,
                    sample=False,
                    previous_obs_action_internal_state=prev_obs_action_internal_state,
                    previous_obs_z_internal_state=prev_obs_z_internal_state,
                )

            # non zero-shot RL methods
            else:
                action = agent.act(
                    observation,
                    sample=False,
                    step=None,
                )

            timestep = env.step(action)
            step += 1

        logger.info("Saving video.")
        video_name = f"{run_name}_{agent.name}_rollout_{task}.mp4"
        video_recorder.save(video_name)
        if self.wandb_logging:
            wandb.log({f"{task}/video": wandb.Video(f"eval_video/{video_name}")})
