import os
import random
from multiprocessing import Manager, Pool
from typing import Any

import gym
import numpy as np
import tqdm
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline.json_writer import JsonWriter

import popgym


class RandomPolicy:
    """A policy that randomly samples actions"""

    def __init__(self, env: gym.Env):
        self.space = env.action_space
        # Cache for faster inference
        self.actions = [self.space.sample() for i in range(10_000)]

    def __call__(self, observation: np.ndarray) -> Any:
        return random.choice(self.actions)


def generate_random_datasets(
    directory: str,
    num_timesteps: int = 10_000_000,
    num_workers: int = 8,
    max_filesize: int = 2 * 1024 * 1024 * 1024,  # 2 GiB
) -> None:
    pool = Pool(num_workers)
    m = Manager()
    q = m.Queue()
    for i in range(1, num_workers + 1):
        q.put(i)
    args = []
    for env, info in popgym.ALL_ENVS.items():
        subdir = info["id"]
        path = f"{directory}/random/{subdir}"
        os.makedirs(path, exist_ok=True)
        instance = env()
        print(f"Generating dataset for {env.__name__}...")
        args.append(
            [path, instance, RandomPolicy(instance), num_timesteps, max_filesize, q]
        )

    pool.starmap(generate_dataset, args)


def generate_dataset(
    directory: str,
    env: gym.Env,
    policy: Any,
    num_timesteps: int,
    max_filesize=2 * 1024 * 1024 * 1024,  # 2 GiB
    queue=None,
    keep_info=False,
) -> None:
    """Generate datasets using a specified policy, exporting datasets to RLlib's
    SampleBatch format as json."""
    builder = SampleBatchBuilder()
    writer = JsonWriter(directory, max_file_size=max_filesize)
    preprocessor = get_preprocessor(env.observation_space)(env.observation_space)
    transition_id = 0
    last_progress_update = 0

    worker_id = queue.get(timeout=10) if queue is not None else 0

    ep_id = 0
    progress = tqdm.tqdm(
        desc=type(env).__name__,
        unit="step",
        position=worker_id,
        total=num_timesteps,
        bar_format="{desc:<30}{percentage:3.0f}%|{bar:10}{r_bar}",
    )
    while True:
        t = 0
        done = False
        obs = env.reset(seed=ep_id, return_info=False)
        prev_action = np.zeros_like(env.action_space.sample())
        prev_reward = 0
        transition_id += 1

        while not done:
            action = policy(obs)
            new_obs, reward, done, info = env.step(action)
            builder.add_values(
                t=t,
                eps_id=ep_id,
                agent_index=0,
                obs=preprocessor.transform(obs),
                action_prob=1.0,
                action_logp=0,
                actions=action,
                rewards=reward,
                prev_action=prev_action,
                prev_rewards=prev_reward,
                dones=done,
                infos=info if keep_info else {},
                new_obs=preprocessor.transform(new_obs),
                transition_id=transition_id,
            )

            obs = new_obs
            prev_action = action
            prev_reward = reward
            t += 1
            transition_id += 1

        ep_id += 1

        if transition_id - last_progress_update > 10_000:
            progress.update(transition_id - last_progress_update)
            last_progress_update = transition_id

        if transition_id > num_timesteps:
            queue.put(worker_id)
            return

        writer.write(builder.build_and_reset())


if __name__ == "__main__":
    generate_random_datasets("/home/popgym/popgym/datasets")
