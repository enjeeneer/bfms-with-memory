# pylint: disable=[protected-access, bare-except, import-outside-toplevel]
"""Utility functions for the project."""
import random
from pathlib import Path

import numpy as np

import pandas as pd
import torch
import yaml
import itertools
# import popgym
# from popgym.wrappers import DiscreteAction
import wandb
import imageio
from google.cloud import storage
from loguru import logger
from os import makedirs
from typing import Optional, List, Union, Any, Dict, Tuple

BASE_DIR = Path(__file__).resolve().parent
GCLOUD_BUCKET = "zero-shot-datasets"
NEW_WANDB_PROJECT = "enjeeneer/zsrl-delta"
OLD_WANDB_PROJECT = "zero-shot-rl/new-beginnings"

EXORL_DOMAINS = [
    "walker",
    "walker_occluded",
    "cheetah",
    "cheetah_occluded",
    "quadruped",
    "quadruped_occluded",
    "point_mass_maze",
    "point_mass_maze_occluded",
    "jaco",
    "jaco_occluded",
]

GOAL_BASED_EXORL_DOMAINS = [
    "point_mass_maze",
    "point_mass_maze_occluded",
    "point_mass_maze_dynamics_occluded",
    "point_mass_maze_rewards_occluded",
    "point_mass_maze_simplified",
]

POPGYM_DOMAINS = [
    "StatelessCartPoleHard",
    "NoisyStatelessCartPoleHard",
    "StatelessPendulumHard",
    "NoisyStatelessPendulumHard",
    "RepeatPreviousHard",
    "RepeatPreviousMedium",
    "RepeatPreviousEasy",
]


def upload_to_gcs_bucket(
    domain_name: str,
    exploration_algorithm: str,
    task: Optional[str] = None,
) -> None:
    """
    Uploads a file to a Google Cloud Storage bucket.
    Args:
        domain_name: The domain name of the dataset.
        exploration_algorithm: The exploration algorithm of the dataset.
        task: The task of the dataset.
    """
    # optimal algo has tasks
    if exploration_algorithm == "optimal":
        gcloud_path = Path(
            f"{domain_name}/{exploration_algorithm}/{task + '/' if task else ''}/"
            f"buffer/dataset.npz"
        )
    else:
        gcloud_path = Path(
            f"{domain_name}/{exploration_algorithm}/{task + '/' if task else ''}/"
            f"buffer/dataset.npz"
        )

    local_path = Path(BASE_DIR / "datasets" / gcloud_path)

    # check if file exists
    if not local_path.exists():
        raise FileNotFoundError(f"File not found at {local_path}.")

    # Upload to GCS
    client = storage.Client(project=GCLOUD_BUCKET)
    client._connection.timeout = 1800
    storage.blob._MAX_MULTIPART_SIZE = 5 * 1024 * 1024  # 5 MB for slower internet
    bucket = client.bucket(GCLOUD_BUCKET)
    blob = bucket.blob(str(gcloud_path))
    blob.chunk_size = 262144 * 4  # change chunksize to 1 MB

    logger.info(
        f"Uploading data to Google Cloud Storage "
        f"bucket at: gs://{GCLOUD_BUCKET}/{gcloud_path}."
    )
    blob.upload_from_filename(local_path, timeout=1800)
    logger.info("Upload complete.")


def download_from_gcs_bucket(
    domain_name: str,
    obs_type: str,
    exploration_algorithm: str,
    task: Optional[str] = None,
    return_dataset: bool = False,
) -> Tuple[Path, Union[None, Dict]]:
    """
    Downloads a file from a Google Cloud Storage bucket.
    Args:
        domain_name: The domain name of the dataset.
        exploration_algorithm: The exploration algorithm of the dataset.
        task: The task of the dataset.
    Returns:
        None
    """
    if obs_type == "pixels":
        dataset_name = "pixels64.pt"
    else:
        dataset_name = "dataset.npz"

    if exploration_algorithm == "rnd" and task is not None:
        gcloud_path = Path(
            f"{domain_name}/{exploration_algorithm}/{task}/buffer/{dataset_name}"
        )
    else:
        gcloud_path = Path(
            f"{domain_name}/{exploration_algorithm}/buffer/{dataset_name}"
        )

    local_path = Path(BASE_DIR / "datasets" / gcloud_path)

    # check if file exists
    if local_path.exists():
        logger.info(f"File already exists at {local_path}.")

        if return_dataset:
            cached_data = {}
            # get transitions
            with local_path.open("rb") as f:
                transitions = torch.load(f)
                cached_data["dataset"] = transitions
            # get task inference data
            task_data_path = local_path.parents[0] / "task_inference_data.pt"
            with task_data_path.open("rb") as f:
                task_inference_data = torch.load(f)
            cached_data = {**cached_data, **task_inference_data}
        else:
            cached_data = None

        return local_path, cached_data

    logger.info(
        f"Downloading data from Google Cloud Storage "
        f"bucket at: gs://{GCLOUD_BUCKET}/{gcloud_path}."
    )

    makedirs(str(local_path.parent), exist_ok=True)

    # Download from GCS
    client = storage.Client()
    bucket = client.bucket(GCLOUD_BUCKET)
    blob = bucket.blob(str(gcloud_path))
    blob.download_to_filename(local_path)

    logger.info("Download complete.")

    return local_path, None


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def pull_model_from_wandb(
    algorithm: str,
    domain_name: str,
    wandb_run_id: str,
    wandb_model_id: str,
    wandb_project: str,
    observation_length: int,
    action_length: int,
    device: torch.device,
    use_local: bool = False,
) -> Union[Any, Dict]:
    """
    Downloads a model from a wandb run and hands weights over to newly
    initialized model. This main use case is for loading models onto
    local CPU that were trained on cloud GPU.
    Args:
        algorithm: algo name
        domain_name: env
        wandb_run_id: wandb run id
        wandb_model_id: name of saved model on wandb run
        wandb_project: wandb project name
        observation_length: env obs length
        action_length: env action length
        device: device to load model onto
    Returns:
        handshaked model: CFB, FB, or CQL agent with weights from wandb run
        config: config dict from wandb run
    """

    # get model from wandb
    logger.info(f"Loading model from: {wandb_project}/runs/{wandb_run_id}")
    api = wandb.Api()
    save_dir = BASE_DIR / "agents" / f"{algorithm}" / "saved_models" / wandb_run_id
    config_path = BASE_DIR / "agents" / f"{algorithm}" / "config.yaml"
    makedirs(str(save_dir), exist_ok=True)
    save_path = save_dir / f"{wandb_model_id}"

    if save_path.exists() and use_local:
        logger.info(f"Model already exists at {save_path}.")

        with open(config_path, "rb") as f:
            config = yaml.safe_load(f)

    else:
        run = api.from_path(f"{wandb_project}/runs/{wandb_run_id}")
        run.file(wandb_model_id).download(root=save_dir.as_posix(), replace=True)
        config = run.config

    # load model
    trained_agent = torch.load(save_path, map_location=torch.device("cpu"))

    if algorithm in ("vcfb", "mcfb", "vcalfb", "mcalfb"):
        from agents.cfb.agent import CFB  # pylint: disable=import-outside-toplevel
        from occlusions import (
            DYNAMICS_OCCLUSIONS,
            REWARD_OCCLUSIONS,
        )  # pylint: disable=import-outside-toplevel

        if "point_mass_maze" in domain_name:
            config["z_dimension"] = 100
            config["discount"] = 0.99

        if algorithm in ("vcfb", "vcalfb"):
            config["vcfb"] = True
            config["mcfb"] = False
        elif algorithm in ("mcfb", "mcalfb"):
            config["vcfb"] = False
            config["mcfb"] = True

        dynamics_occlusion = DYNAMICS_OCCLUSIONS[domain_name](
            obs_type=config["obs_type"], frames=config["frames"]
        )
        rewards_occlusion = REWARD_OCCLUSIONS[config["domain_name"]](
            obs_type=config["obs_type"], frames=config["frames"]
        )

        handshake_agent = CFB(
            observation_length=dynamics_occlusion.observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=config["preprocessor_hidden_dimension"],
            preprocessor_output_dimension=config["preprocessor_output_dimension"],
            preprocessor_hidden_layers=config["preprocessor_hidden_layers"],
            forward_hidden_dimension=config["forward_hidden_dimension"],
            forward_hidden_layers=config["forward_hidden_layers"],
            backward_hidden_dimension=config["backward_hidden_dimension"],
            backward_hidden_layers=config["backward_hidden_layers"],
            actor_hidden_dimension=config["actor_hidden_dimension"],
            actor_hidden_layers=config["actor_hidden_layers"],
            preprocessor_activation=config["preprocessor_activation"],
            forward_activation=config["forward_activation"],
            backward_activation=config["backward_activation"],
            actor_activation=config["actor_activation"],
            z_dimension=config["z_dimension"],
            actor_learning_rate=config["actor_learning_rate"],
            critic_learning_rate=config["critic_learning_rate"],
            learning_rate_coefficient=config["learning_rate_coefficient"],
            orthonormalisation_coefficient=config["orthonormalisation_coefficient"],
            discount=config["discount"],
            batch_size=config["batch_size"],
            z_mix_ratio=config["z_mix_ratio"],
            gaussian_actor=config["gaussian_actor"],
            std_dev_clip=config["std_dev_clip"],
            std_dev_schedule=config["std_dev_schedule"],
            tau=config["tau"],
            device=device,
            total_action_samples=config["total_action_samples"],
            ood_action_weight=config["ood_action_weight"],
            alpha=config["alpha"],
            target_conservative_penalty=config["target_conservative_penalty"],
            vcfb=config["vcfb"],
            mcfb=config["mcfb"],
            dvcfb=False,
            lagrange=config["lagrange"],
            layernorm=True,
            goal_dimension=rewards_occlusion.observation_length,
        )

        handshake_agent.FB.load_state_dict(trained_agent.FB.state_dict())
        handshake_agent.actor.load_state_dict(trained_agent.actor.state_dict())

    elif algorithm in ("fb", "fb-probe"):
        from agents.fb.agent import FB  # pylint: disable=import-outside-toplevel
        from occlusions import (
            DYNAMICS_OCCLUSIONS,
            REWARD_OCCLUSIONS,
        )  # pylint: disable=import-outside-toplevel

        if domain_name in (
            "point_mass_maze",
            "point_mass_maze_simplified",
            "point_mass_maze_hard",
            "point_mass_maze_occluded",
        ):
            config["discount"] = 0.99
            config["z_dimension"] = 100

        dynamics_occlusion = DYNAMICS_OCCLUSIONS[domain_name](
            obs_type=config["obs_type"], frames=config["frames"]
        )
        rewards_occlusion = REWARD_OCCLUSIONS[config["domain_name"]](
            obs_type=config["obs_type"], frames=config["frames"]
        )

        handshake_agent = FB(
            observation_dims=dynamics_occlusion.observation_length,
            action_length=action_length,
            forward_hidden_dimension=config["forward_hidden_dimension"],
            forward_hidden_layers=config["forward_hidden_layers"],
            backward_hidden_dimension=config["backward_hidden_dimension"],
            preprocessed_dimension=config["preprocessed_dimension"],
            postprocessed_dimension=config["postprocessed_dimension"],
            backward_hidden_layers=config["backward_hidden_layers"],
            actor_hidden_dimension=config["actor_hidden_dimension"],
            actor_hidden_layers=config["actor_hidden_layers"],
            goal_dimension=rewards_occlusion.observation_length,
            forward_activation=config["forward_activation"],
            backward_activation=config["backward_activation"],
            actor_activation=config["actor_activation"],
            z_dimension=config["z_dimension"],
            actor_learning_rate=config["actor_learning_rate"],
            critic_learning_rate=config["critic_learning_rate"],
            learning_rate_coefficient=config["learning_rate_coefficient"],
            orthonormalisation_coefficient=config["orthonormalisation_coefficient"],
            discount=config["discount"],
            batch_size=config["batch_size"],
            z_mix_ratio=config["z_mix_ratio"],
            gaussian_actor=config["gaussian_actor"],
            std_dev_clip=config["std_dev_clip"],
            std_dev_schedule=config["std_dev_schedule"],
            tau=config["tau"],
            device=device,
            name=config["name"],
            exploration_epsilon=config["exploration_epsilon"]
            if "exploration_epsilon" in config
            else 0.0,
            boltzmann_temperature=config["boltzmann_temperature"]
            if "boltzmann_temperature" in config
            else 0.0,
            layernorms=True,
            discrete_actions=False,
            observation_type="states",
            frames=1,
            z_inference_steps=config["z_inference_steps"],
        )

        handshake_agent.FB.load_state_dict(trained_agent.FB.state_dict())
        handshake_agent.actor.load_state_dict(trained_agent.actor.state_dict())


def pull_exorl_data(
    run_path: str,
    dynamics: Optional[List[float]],
    tasks: Optional[List[str]] = None,
    step_samples: Optional[int] = None,
    algorithm: Optional[str] = None,
    env_gen: bool = False,
    gen: bool = False
):
    api = wandb.Api(timeout=60)
    run = api.run(run_path)
    keys = []

    logger.info(f"Pulling data from wandb run: {run_path}.")
    if step_samples is not None:
        data = run.history(keys=keys, samples=step_samples)
    else:
        if algorithm == "td3" and tasks is not None:
            if dynamics is not None:
                keys = [
                    f"{run.config['eval_tasks'][0]}/mass={multiplier}x;"
                    f" damping={multiplier}x"
                    for multiplier in dynamics
                ]
            data = run.history(keys=keys)
        elif algorithm != "td3" and tasks is not None:
            if dynamics is not None:
                for task in tasks:
                    if run.config["train_multipliers"] == [1]:
                        keys.append(
                            [f"{task}/mass=1.0x; damping=1.0x"]
                        )
                    elif run.config["train_multipliers"] == [2]:
                        keys.append(
                            [f"{task}/mass=2.0x; damping=2.0x"]
                        )
                    else:
                        for multiplier in dynamics:
                            keys.append(
                                [f"{task}/mass={multiplier}x; damping={multiplier}x"]
                            )
            # keys.append("task_reward_iqm")
            keys = list(itertools.chain(*keys))
            data = run.history(keys=keys)
        else:
            data = run.history(keys=keys)

    data = data.set_index("_step").astype(int)

    column_mapper = {}
    if run.config["algorithm"] in ["fb"]:
        try:
            stack = run.config["frames"] == 4
            if stack:
                memory_name = "stack"
            else:
                if gen:
                    if len(run.config["train_multipliers"]) == 1:
                        memory_name = "MDP"
                    else:
                        memory_name = "none"
                else:
                    memory_name = run.config["memory_type"]
        except:
            memory_name = "none"

    elif run.config["algorithm"] in ["sf-hilp"]:
        memory_name = "stack"
    elif run.config["algorithm"] in ["td3gru"]:
        memory_name = "gru"
    else:
        memory_name = run.config["memory_type"]

    for col in data.columns:

        # search for task in column name
        for task_name in keys:
            if task_name in col:
                if env_gen:
                    col_name = (
                        run.config["algorithm"],
                        memory_name,
                        task_name,
                        run.config["seed"],
                    )
                else:
                    col_name = (
                        run.config["algorithm"],
                        memory_name,
                        run.config["domain_name"],
                        task_name,
                        run.config["seed"],
                    )

                column_mapper[col] = col_name

    data = data.rename(columns=column_mapper)
    if not env_gen:
        data.columns = pd.MultiIndex.from_tuples(
            data.columns,
            names=["algorithm", "memory_type", "domain_name", "task", "seed"],
        )
    else:
        data.columns = pd.MultiIndex.from_tuples(
            data.columns,
            names=["algorithm", "memory_type", "task", "seed"],
        )

    return data


def pull_popgym_data(
    run_path: str,
    steps: int = 500e3,
):
    api = wandb.Api(timeout=60)
    run = api.run(run_path)
    key = ["iqm_reward"]

    logger.info(f"Pulling data from wandb run: {run_path}.")
    data = run.history(keys=key)
    data = data.set_index("_step")

    # take first 500k steps
    data = data.loc[:steps]

    column_mapper = {}
    if run.config["algorithm"] in ["fb"]:
        memory_name = "stack"
    elif run.config["algorithm"] in ["td3gru"]:
        memory_name = "gru"
    else:
        memory_name = run.config["memory_type"]

    for col in data.columns:
        col_name = (
            run.config["algorithm"],
            memory_name,
            run.config["domain_name"],
            run.config["seed"],
        )

        column_mapper[col] = col_name

    data = data.rename(columns=column_mapper)
    data.columns = pd.MultiIndex.from_tuples(
        data.columns,
        names=["algorithm", "memory_type", "domain_name", "seed"],
    )

    # replace nans with 0
    data = data.fillna(0)

    return data


def pull_wandb_data(
    run_path: str,
    keys: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    hyperparam: Optional[str] = None,
    step_samples: Optional[int] = None,
    algorithm: Optional[str] = None,
):
    api = wandb.Api()
    run = api.run(run_path)

    logger.info(f"Pulling data from Wandb run: {run_path}.")
    if step_samples is not None:
        data = run.history(keys=keys, samples=step_samples)
    else:
        if algorithm == "td3" and tasks is not None:
            key = [f"eval/{run.config['eval_tasks'][0]}/episode_reward_iqm"]
            data = run.history(keys=key)
        elif algorithm == "gciql":
            data = run.history(keys=keys)
        elif algorithm != "td3" and tasks is not None:
            keys = [f"eval/{task}/episode_reward_iqm" for task in tasks]
            keys.append("eval/task_reward_iqm")
            data = run.history(keys=keys)
        else:
            data = run.history(keys=keys)

    data = data.set_index("_step").astype(int)
    # rename gciql cols
    if algorithm == "gciql":
        # extract the string before the first /
        old_tasks = [col.split("/")[0] for col in data.columns]
        new_cols = []
        for task in old_tasks:
            if task != "all_tasks":
                new_cols.append(f"eval/{task}/episode_reward_iqm")
            else:
                new_cols.append("eval/task_reward_iqm")
        data.columns = new_cols

    if hyperparam is not None:
        col_names = [
            (
                col,
                run.config[hyperparam],
                run.config["exploration_algorithm"],
            )
            for col in data.columns
        ]

        data.columns = pd.MultiIndex.from_tuples(
            col_names, names=["stat", hyperparam, "exploration_algorithm"]
        )

    if tasks is not None:
        column_mapper = {}
        for col in data.columns:

            # search for task in column name
            for task_name in tasks:
                if task_name in col:

                    col_name = (
                        run.config["algorithm"],
                        run.config["exploration_algorithm"],
                        run.config["domain_name"],
                        task_name,
                        run.config["seed"],
                    )

                    column_mapper[col] = col_name

        column_mapper["eval/task_reward_iqm"] = (
            run.config["algorithm"],
            run.config["exploration_algorithm"],
            run.config["domain_name"],
            "all_tasks",
            run.config["seed"],
        )

        data = data.rename(columns=column_mapper)
        data.columns = pd.MultiIndex.from_tuples(
            data.columns,
            names=["algorithm", "exploration_algorithm", "domain_name", "task", "seed"],
        )

    logger.info("Pull complete.")

    return data


class VideoRecorder:
    """
    Records videos of agents performing tasks in the DM
    control suite.
    """

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]],
        render_size: int = 256,
        fps: int = 20,
        camera_id: int = 0,
        use_wandb: bool = False,
    ) -> None:
        self.save_dir: Optional[Path] = None
        if root_dir is not None:
            self.save_dir = Path(root_dir) / "eval_video"
            self.save_dir.mkdir(exist_ok=True)
        self.enabled = False
        self.render_size = render_size
        self.fps = fps
        self.frames: List[np.ndarray] = []
        self.camera_id = camera_id
        self.use_wandb = use_wandb

    def init(self, env, enabled: bool = True) -> None:
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env) -> None:
        if self.enabled:
            if hasattr(env, "physics"):
                if env.physics is not None:
                    frame = env.physics.render(
                        height=self.render_size,
                        width=self.render_size,
                        camera_id=self.camera_id,
                    )
                else:
                    frame = env.base_env.render()
            else:
                frame = env.render()
            self.frames.append(frame)

    def log_to_wandb(self) -> None:
        frames = np.transpose(np.array(self.frames), (0, 3, 1, 2))
        fps, skip = 6, 8
        wandb.log(
            {
                "eval/video": wandb.Video(
                    frames[::skip, :, ::2, ::2], fps=fps, format="gif"
                )
            }
        )

    def save(self, file_name: str) -> None:
        if self.enabled:
            if self.use_wandb:
                self.log_to_wandb()
            assert self.save_dir is not None
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)  # type: ignore

