# pylint: disable=protected-access

"""Evaluates the performance of pre-trained agents."""
import yaml
import torch
import argparse
from argparse import ArgumentParser
import datetime
import numpy as np
import os
from pathlib import Path

from agents.base import (
    MemoryEfficientOfflineReplayBuffer,
)
from agents.workspaces import ExorlWorkspace
from agents.fb_m.agent import MemoryBasedFB
from agents.hilp_m.agent import MemoryBasedHILP

from rewards import RewardFunctionConstructor
from occlusions import DYNAMICS_OCCLUSIONS, REWARD_OCCLUSIONS
from utils import set_seed_everywhere, download_from_gcs_bucket, BASE_DIR

parser = ArgumentParser()
parser.add_argument("algorithm", type=str)
parser.add_argument("domain_name", type=str)
parser.add_argument("exploration_algorithm", type=str)
parser.add_argument(
    "--obs_type", type=str, default="states", choices=["pixels", "states"]
)
parser.add_argument(
    "--wandb_logging", default=True, action=argparse.BooleanOptionalAction
)
parser.add_argument("--wandb_entity", type=str)
parser.add_argument("--wandb_project", type=str)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train_multipliers", nargs="+", default=[1.0])
parser.add_argument("--eval_multipliers", nargs="+", default=[1.0])
parser.add_argument("--train_task", type=str)
parser.add_argument("--episodes", type=int, default=5000)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--dataset_transitions", type=int, default=5000000)
parser.add_argument("--eval_tasks", nargs="+", required=True)
parser.add_argument("--learning_steps", type=int, default=1000000)
parser.add_argument(
    "--save_model", default=False, action=argparse.BooleanOptionalAction
)
parser.add_argument("--memory_type", type=str)
parser.add_argument("--transformer_dimension", type=int, default=32)
parser.add_argument("--gru_dimension", type=int, default=512)
parser.add_argument("--s4_dimension", type=int, default=32)
parser.add_argument("--transformer_heads", type=int, default=4)
parser.add_argument("--num_encoder_layers", type=int, default=1)
parser.add_argument("--transformer_attention", type=str, default="flash")
parser.add_argument("--frames", type=int, default=None)
parser.add_argument("--goal_frames", type=int, default=1)
parser.add_argument("--flickering_prob", type=float)
parser.add_argument("--noise_std", type=float)
parser.add_argument("--missing_sensor_prob", type=float)
parser.add_argument("--z_inference_steps", type=int, default=10000)
parser.add_argument(
    "--pad_with_zeros", default=False, action=argparse.BooleanOptionalAction
)
parser.add_argument(
    "--inference_memory", default=False, action=argparse.BooleanOptionalAction
)
parser.add_argument("--pooling", type=str, default="last")
parser.add_argument(
    "--actor_obs_z_encoder", default=False, action=argparse.BooleanOptionalAction
)
parser.add_argument("--history_length", type=int, default=1)
parser.add_argument("--backward_history_length", type=int, default=0)
parser.add_argument(
    "--nonlinear_z", default=False, action=argparse.BooleanOptionalAction
)
parser.add_argument(
    "--recurrent_B", default=False, action=argparse.BooleanOptionalAction
)
parser.add_argument(
    "--recurrent_F", default=False, action=argparse.BooleanOptionalAction
)
parser.add_argument(
    "--gradient_clipping", default=True, action=argparse.BooleanOptionalAction
)
parser.add_argument(
    "--action_in_B", default=False, action=argparse.BooleanOptionalAction
)
parser.add_argument("--reward_scale", type=int, default=1.0)
parser.add_argument("--z_embedding_dimension", type=int, default=128)
args = parser.parse_args()

# set wandb save dir
os.environ["WANDB_DATA_DIR"] = str(BASE_DIR)
os.environ["GOOGLE_CLOUD_PROJECT"] = "zero-shot-datasets"

assert args.episodes * 1000 >= args.dataset_transitions

if args.algorithm == "fb_m" and args.history_length == 0:
    raise ValueError("MemoryBasedFB requires history_length > 0")

if args.dataset_transitions == -1:
    args.dataset_transitions = None

working_dir = Path.cwd()

if args.algorithm in ("sf-lap", "sf-hilp"):
    algo_dir = "sf"
    config_path = working_dir / "agents" / algo_dir / "config.yaml"
    model_dir = working_dir / "agents" / algo_dir / "saved_models"
else:
    config_path = working_dir / "agents" / args.algorithm / "config.yaml"
    model_dir = working_dir / "agents" / args.algorithm / "saved_models"

with open(config_path, "rb") as f:
    config = yaml.safe_load(f)

time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

config.update(vars(args))

if config["device"] is None:
    config["device"] = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

# correct multipliers dtype
config["train_multipliers"] = [float(i) for i in config["train_multipliers"]]
config["eval_multipliers"] = [float(i) for i in config["eval_multipliers"]]

# less frequent evals for maze it contains many more tasks
if "maze" in config["domain_name"]:
    config["eval_frequency"] = config["eval_frequency"] * 5

if args.save_model:
    # at the moment we only support saving w.r.t. performance on one eval task
    # assert len(args.eval_body_mass_multipliers) == 1
    assert (
        1 in config["eval_multipliers"]
    ), "1 must be in eval_multipliers to save model"

set_seed_everywhere(config["seed"])

if "simplified" in config["domain_name"]:
    MDP_domain_name = config["domain_name"].replace("_simplified", "")
elif "hard" in config["domain_name"]:
    MDP_domain_name = config["domain_name"].replace("_hard", "")
elif any(
    substring in config["domain_name"]
    for substring in ["occluded", "flickering", "noise", "sensors"]
):
    if "walker" in config["domain_name"]:
        MDP_domain_name = "walker"
    elif "cheetah" in config["domain_name"]:
        MDP_domain_name = "cheetah"
    elif "quadruped" in config["domain_name"]:
        MDP_domain_name = "quadruped"
    elif "jaco" in config["domain_name"]:
        MDP_domain_name = "jaco"
    elif "point_mass_maze" in config["domain_name"]:
        MDP_domain_name = "point_mass_maze"
else:
    MDP_domain_name = config["domain_name"]

# pull data from GCS
dataset_paths = []
if config["algorithm"] in (
    "fb_m",
    "hilp_m",
):
    for multiplier in config["train_multipliers"]:
        dataset_path, pixel_dataset = download_from_gcs_bucket(
            domain_name=MDP_domain_name,
            exploration_algorithm=config["exploration_algorithm"],
            task=f"mass={multiplier}x;" f"damping={multiplier}x",
            obs_type=config["obs_type"],
            return_dataset=config["obs_type"] == "pixels",
        )
        dataset_paths.append(dataset_path)
    relabel = False

else:
    for multiplier in config["train_multipliers"]:
        dataset_path, pixel_dataset = download_from_gcs_bucket(
            domain_name=MDP_domain_name,
            exploration_algorithm=config["exploration_algorithm"],
            task=f"mass={multiplier}x;" f"damping={multiplier}x",
            obs_type=config["obs_type"],
        )
        dataset_paths.append(dataset_path)
    relabel = True

# create reward function constructors for envs with different physics
reward_constructors = {}
for multiplier in config["eval_multipliers"]:
    reward_constructor = RewardFunctionConstructor(
        domain_name=MDP_domain_name,
        task_names=config["eval_tasks"],
        seed=config["seed"],
        device=config["device"],
        body_mass_multiplier=multiplier,
        damping_multiplier=multiplier,
        obs_type=config["obs_type"],
        frames=config["frames"],
    )
    reward_constructors[(multiplier, multiplier)] = reward_constructor

# extract reward constructor for unmodified physics
train_reward_constructor = RewardFunctionConstructor(
    domain_name=MDP_domain_name,
    task_names=config["eval_tasks"],
    seed=config["seed"],
    device=config["device"],
    body_mass_multiplier=config["train_multipliers"][0],
    damping_multiplier=config["train_multipliers"][0],
    obs_type=config["obs_type"],
    frames=config["frames"],
)
(
    observation_spec,
    action_length,
    action_range,
) = train_reward_constructor.get_metadata()

# dynamics and reward occlusions
dynamics_occlusion = DYNAMICS_OCCLUSIONS[config["domain_name"]](
    domain=MDP_domain_name,
    obs_type=config["obs_type"],
    frames=config["frames"],
    flickering_prob=config["flickering_prob"],
    noise_std=config["noise_std"],
    missing_sensor_prob=config["missing_sensor_prob"],
)
rewards_occlusion = REWARD_OCCLUSIONS[config["domain_name"]](
    domain=MDP_domain_name,
    obs_type=config["obs_type"],
    frames=config["frames"],
    flickering_prob=config["flickering_prob"],
    noise_std=config["noise_std"],
    missing_sensor_prob=config["missing_sensor_prob"],
)

if config["algorithm"] == "fb_m":
    if config["memory_type"] in ["transformer"] and config["pooling"] is None:
        raise ValueError("FART/Transformer requires pooling to be set")

    if (
        config["memory_type"] == "mlp"
        and config["recurrent_F"]
        and config["inference_memory"] is False
    ):
        raise ValueError("MLP (frame-stacking) requires inference_memory to be True")

    if config["recurrent_B"] and config["backward_history_length"] == 0:
        raise ValueError("recurrent B requires backward_history_length > 0")

    if not config["recurrent_F"] and config["history_length"] != 1:
        raise ValueError(
            "If you are not using recurrent F then history_length must be 1."
        )

    if config["recurrent_F"] and config["history_length"] <= 1:
        raise ValueError(
            "The 'history_length' hyperparameter includes the current observation"
            "so to have a non-zero history length, it must be set to at least 2 for ."
        )

    agent = MemoryBasedFB(
        observation_dims=dynamics_occlusion.observation_length,
        observation_type=config["obs_type"],
        action_length=action_length,
        goal_dimension=rewards_occlusion.observation_length,
        goal_frames=config["goal_frames"],
        forward_hidden_dimension=config["forward_hidden_dimension"],
        forward_hidden_layers=config["forward_hidden_layers"],
        preprocessed_dimension=config["preprocessed_dimension"],
        postprocessed_dimension=config["postprocessed_dimension"],
        backward_hidden_dimension=config["backward_hidden_dimension"],
        backward_hidden_layers=config["backward_hidden_layers"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        obs_encoder_hidden_dimension=config["obs_encoder_hidden_dimension"],
        forward_activation=config["forward_activation"],
        backward_activation=config["backward_activation"],
        z_dimension=config["z_dimension"],
        critic_learning_rate=config["critic_learning_rate"],
        actor_learning_rate=config["actor_learning_rate"],
        learning_rate_coefficient=config["learning_rate_coefficient"],
        orthonormalisation_coefficient=config["orthonormalisation_coefficient"],
        discount=config["discount"],
        recurrent_F=config["recurrent_F"],
        recurrent_B=config["recurrent_B"],
        batch_size=config["batch_size"],
        z_mix_ratio=config["z_mix_ratio"],
        std_dev_schedule=config["std_dev_schedule"],
        tau=config["tau"],
        device=config["device"],
        name=config["name"],
        memory_type=config["memory_type"],
        history_length=config["history_length"],
        backward_history_length=config["backward_history_length"],
        std_dev_clip=config["std_dev_clip"],
        z_inference_steps=config["z_inference_steps"],
        pooling=config["pooling"],
        inference_memory=config["inference_memory"],
        transformer_dimension=config["transformer_dimension"],
        s4_dimension=config["s4_dimension"],
        transformer_n_heads=config["transformer_heads"],
        num_encoder_layers=config["num_encoder_layers"],
        transformer_attention=config["transformer_attention"],
        gradient_clipping=config["gradient_clipping"],
        actor_obs_z_encoder=config["actor_obs_z_encoder"],
        boltzmann_temperature=config["boltzmann_temperature"],
        gru_dimension=config["gru_dimension"],
    )

    # load buffer
    # if we're working with pixels then we init an empty buffer
    # and load in the pre-formed dataset
    if config["obs_type"] == "pixels":
        replay_buffer = MemoryEfficientOfflineReplayBuffer(
            dataset_paths=dataset_paths,
            discount=config["discount"],
            device=config["device"],
            max_episodes=config["episodes"],
            reward_occlusion=rewards_occlusion,
            dynamics_occlusion=dynamics_occlusion,
            relabel=relabel,
            frames=config["frames"],
            history_length=config["history_length"],
            goal_history_length=config["backward_history_length"],
            obs_type=config["obs_type"],
            reward_constructors=reward_constructors,
            eval_multipliers=config["eval_multipliers"],
            load_on_init=False,
            pad_with_zeros=config["pad_with_zeros"],
            goal_frames=config["goal_frames"],
        )

        replay_buffer._storage = pixel_dataset["dataset"]
        replay_buffer._max_episodes = replay_buffer._storage["pixel"].shape[0]
        replay_buffer._episode_lengths = np.repeat(
            replay_buffer._storage["pixel"].shape[1] - 1, replay_buffer._max_episodes
        )
        replay_buffer._full = True

        # hack the dones (we know last transition is terminal)
        not_dones = np.ones_like(replay_buffer._storage["discount"], dtype=float)
        not_dones[:, -1] = 0.0
        replay_buffer._storage["not_done"] = not_dones

        # pre-computed task inference data comes in
        # [z_inference_steps, 3 x RGB frames, ...]
        # only take final frame
        goals_z = {}
        for multiplier, data in pixel_dataset[MDP_domain_name]["goals_z"].items():
            goals_z[multiplier] = data[:, -3:, ...]

        replay_buffer._storage["goals_z"] = goals_z
        replay_buffer._storage["rewards_z"] = pixel_dataset[MDP_domain_name][
            "rewards_z"
        ]
    else:
        replay_buffer = MemoryEfficientOfflineReplayBuffer(
            dataset_paths=dataset_paths,
            discount=config["discount"],
            device=config["device"],
            max_episodes=config["episodes"],
            reward_occlusion=rewards_occlusion,
            dynamics_occlusion=dynamics_occlusion,
            relabel=relabel,
            frames=config["frames"],
            history_length=config["history_length"],
            goal_history_length=config["backward_history_length"],
            obs_type=config["obs_type"],
            reward_constructors=reward_constructors,
            eval_multipliers=config["eval_multipliers"],
            load_on_init=True,
            pad_with_zeros=config["pad_with_zeros"],
            goal_frames=config["goal_frames"],
        )

    z_inference_steps = config["z_inference_steps"]
    train_std = config["std_dev_schedule"]
    eval_std = config["std_dev_eval"]

elif config["algorithm"] in ["hilp_m"]:

    config["sf_features"] = "hilp"
    config["hilp_p_random_goal"] = 0.375
    if MDP_domain_name == "walker":
        config["hilp_discount"] = 0.96

    agent = MemoryBasedHILP(
        observation_dims=dynamics_occlusion.observation_length,
        action_length=action_length,
        observation_type=config["obs_type"],
        goal_dimension=rewards_occlusion.observation_length,
        preprocessed_dimension=config["preprocessed_dimension"],
        postprocessed_dimension=config["postprocessed_dimension"],
        forward_hidden_dimension=config["forward_hidden_dimension"],
        forward_hidden_layers=config["forward_hidden_layers"],
        features_hidden_dimension=config["features_hidden_dimension"],
        features_hidden_layers=config["features_hidden_layers"],
        features_activation=config["features_activation"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        forward_activation=config["forward_activation"],
        actor_activation=config["actor_activation"],
        z_dimension=config["z_dimension"],
        sf_learning_rate=config["sf_learning_rate"],
        feature_learning_rate=config["feature_learning_rate"],
        actor_learning_rate=config["actor_learning_rate"],
        batch_size=config["batch_size"],
        gaussian_actor=config["gaussian_actor"],
        std_dev_clip=config["std_dev_clip"],
        std_dev_schedule=config["std_dev_schedule"],
        z_inference_steps=config["z_inference_steps"],
        tau=config["tau"],
        device=config["device"],
        name=config["name"],
        z_mix_ratio=config["z_mix_ratio"],
        q_loss=MDP_domain_name in ("quadruped", "jaco")
        if config["sf_features"] == "hilp"
        else True,
        features=config["sf_features"],
        hilp_discount=config["hilp_discount"],
        hilp_iql_expectile=config["hilp_iql_expectile"],
        frames=config["frames"],
        memory_type=config["memory_type"],
        history_length=config["history_length"],
        phi_history_length=config["backward_history_length"],
        obs_encoder_hidden_dimension=config["obs_encoder_hidden_dimension"],
        pooling=config["pooling"],
        inference_memory=config["inference_memory"],
        transformer_dimension=config["transformer_dimension"],
        s4_dimension=config["s4_dimension"],
        transformer_n_heads=config["transformer_heads"],
        num_encoder_layers=config["num_encoder_layers"],
        transformer_attention=config["transformer_attention"],
        gru_dimension=config["gru_dimension"],
        recurrent_F=config["recurrent_F"],
        recurrent_phi=config["recurrent_B"],
    )

    # load buffer
    replay_buffer = MemoryEfficientOfflineReplayBuffer(
        dataset_paths=dataset_paths,
        discount=config["discount"],
        device=config["device"],
        max_episodes=config["episodes"],
        reward_occlusion=rewards_occlusion,
        dynamics_occlusion=dynamics_occlusion,
        relabel=relabel,
        frames=config["frames"],
        history_length=config["history_length"],
        goal_history_length=config["backward_history_length"],
        obs_type=config["obs_type"],
        reward_constructors=reward_constructors,
        eval_multipliers=config["eval_multipliers"],
        load_on_init=True,
        pad_with_zeros=config["pad_with_zeros"],
        goal_frames=config["goal_frames"],
    )

    z_inference_steps = config["z_inference_steps"]
    train_std = config["std_dev_schedule"]
    eval_std = config["std_dev_eval"]
else:
    raise NotImplementedError(f"Algorithm {config['algorithm']} not implemented")

workspace = ExorlWorkspace(
    reward_constructors=reward_constructors,
    learning_steps=config["learning_steps"],
    model_dir=model_dir,
    goal_frames=config["goal_frames"],
    eval_frequency=config["eval_frequency"],
    eval_rollouts=config["eval_rollouts"],
    z_inference_steps=z_inference_steps,
    train_std=train_std,
    eval_std=eval_std,
    wandb_logging=config["wandb_logging"],
    device=config["device"],
    eval_multipliers=config["eval_multipliers"],
    save_model=config["save_model"],
    save_multiplier=config["eval_multipliers"][0] if config["save_model"] else None,
    reward_occlusion=rewards_occlusion,
    dynamics_occlusion=dynamics_occlusion,
    wandb_project=config["wandb_project"],
    wandb_entity=config["wandb_entity"],
)

if __name__ == "__main__":
    workspace.train(
        agent=agent,
        agent_config=config,
        replay_buffer=replay_buffer,
    )
