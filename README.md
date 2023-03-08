# POPGym: Partially Observable Process Gym
![tests](https://github.com/smorad/popgym/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/smorad/popgym/branch/master/graph/badge.svg?token=I47IDFZXSV)](https://codecov.io/gh/smorad/popgym)

POPGym is designed to benchmark memory in deep reinforcement learning. It contains a set of [environments](#popgym-environments) and a collection of [memory model baselines](#popgym-baselines). The full paper is available on [OpenReview](https://openreview.net/forum?id=chDrutUTs0K).


## Table of Contents
1. [POPGym Environments](#popgym-environments)
    1. [Setup](#setup)
    2. [Usage](#usage)
    3. [Table of Environments](#table-of-environments)
    4. [Environment Descriptions](#environment-descriptions)
2. [POPGym Baselines](#popgym-baselines)
    1. [Setup](#setup-1)
    2. [Usage](#usage-1)
    3. [Available Baselines](#available-baselines)
3. [Leaderboard](#leaderboard)
4. [Contributing](#contributing)
5. [Citing](#citing)

## POPGym Environments

POPGym contains Partially Observable Markov Decision Process (POMDP) environments following the [Openai Gym](https://github.com/openai/gym) interface. Our environments follow a few basic tenets:

1. **Painless Setup** - `popgym` environments require only `gymnasium`, `numpy`, and `mazelib` as dependencies
2. **Laptop-Sized Tasks** - Most tasks can be solved in less than a day on the CPU 
3. **True Generalization** - All environments are heavily randomized.


### Setup 
<details><summary>Expand</summary>
<p>

You may install `popgym` via `pip` or from source.
#### Pip
```bash
# Works with python <= 3.10 due to mazelib dependency
pip install popgym
```

#### From Source
To install the environments:
```bash
git clone https://github.com/smorad/popgym
cd popgym
pip install .
```
</p>
</details>

### Usage

<details><summary>Expand</summary>
<p>

```python
import gymnasium as gym
import popgym
from popgym.wrappers import PreviousAction, Antialias, Markovian
from popgym.core.observability import Observability, STATE
# List all envs, see popgym/__init__.py 
env_classes = popgym.ALL_ENVS.keys()
print(env_classes)
env_names = [e["id"] for e in popgym.ALL_ENVS.values()]
print(env_names)
# Create env
env = popgym.envs.stateless_cartpole.StatelessCartPoleEasy()
# In POMDPs, we often condition on the last action along with the observation.
# We can do this using the PreviousAction wrapper.
wrapped_env = PreviousAction(env)
# To prevent observation aliasing during the first timestep of
# each episode (where the previous action is undefined), we can also 
# combine the PreviousAction wrapper with the Antialias wrapper
wrapped_env = Antialias(wrapped_env)
# Finally, we can decide if we want the hidden Markov state.
# This can be part of the observation, placed in the info dict, etc.
wrapped_env = Markovian(wrapped_env, Observability.FULL_IN_INFO_DICT)

wrapped_env.reset()
obs, reward, terminated, truncated, info = wrapped_env.step(wrapped_env.action_space.sample())
print(obs)
# Outputs:
# (
  ## Original observation
  # array([0.0348076 , 0.02231686], dtype=float32), 
  ## Previous action
  # 1, 
  ## Is initial timestep (antialias)
  #0
#)

# Print the hidden Markov state
print(info[STATE])
# Outputs:
# array([ 0.0348076 ,  0.14814377,  0.02231686, -0.31778395], dtype=float32)
```
</p>
</details>

### Table of Environments

<details><summary>Expand</summary>
<p>

| Environment                                                                                             |         Tags      | Temporal Ordering | Colab FPS         | Macbook Air (2020) FPS    |
|---------------------------------------------------------------------------------------------------------|-------------------|-------------------|-------------------|---------------------------|
| [Battleship](#battleship) [(Code)](popgym/envs/battleship.py)                                           |Game               |None               |  117,158          |  235,402                  |
| [Concentration](#concentration) [(Code)](popgym/envs/concentration.py)                                  |Game               |Weak               |  47,515           |  157,217                  |
| [Higher Lower](#higher-lower) [(Code)](popgym/envs/higher_lower.py)                                     |Game, Noisy        |None               |  24,312           |  76,903                   |
| [Labyrinth Escape](#labyrinth-escape) [(Code)](popgym/envs/labyrinth_escape.py)                         |Navigation         |Strong             |  1,399            |  41,122                   |
| [Labyrinth Explore](#labyrinth-explore) [(Code)](popgym/envs/labyrinth_explore.py)                      |Navigation         |Strong             |  1,374            |  30,611                   |
| [Minesweeper](#minesweeper) [(Code)](popgym/envs/minesweeper.py)                                        |Game               |None               |  8,434            |  32,003                   |
| [Multiarmed Bandit](#multiarmed-bandit) [(Code)](popgym/envs/multiarmed_bandit.py)                      |Noisy              |None               |  48,751           |  469,325                  |
| [Autoencode](#autoencode) [(Code)](popgym/envs/autoencode.py)                                           |Diagnostic         |Strong             |  121,756          |  251,997                  |
| [Count Recall](#count-recall) [(Code)](popgym/envs/count_recall.py)                                     |Diagnostic, Noisy  |None               |  16,799           |  50,311                   |
| [Repeat First](#repeat-first) [(Code)](popgym/envs/repeat_first.py)                                     |Diagnostic         |None               |  23,895           |  155,201                  |
| [Repeat Previous](#repeat-previous) [(Code)](popgym/envs/repeat_previous.py)                            |Diagnostic         |Strong             |  50,349           |  136,392                  |
| [Stateless Cartpole](#stateless-cartpole) [(Code)](popgym/envs/stateless_cartpole.py)                   |Control            |Strong             |  73,622           |  218,446                  |
| [Noisy Stateless Cartpole](#noisy-stateless-cartpole) [(Code)](popgym/envs/noisy_stateless_cartpole.py) |Control, Noisy     |Strong             |  6,269            |  66,891                   |
| [Stateless Pendulum](#noisy-stateless-pendulum) [(Code)](popgym/envs/stateless_pendulum.py)             |Control            |Strong             |  8,168            |  26,358                   |
| [Noisy Stateless Pendulum](#noisy-stateless-pendulum) [(Code)](popgym/envs/noisy_stateless_pendulum.py) |Control, Noisy     |Strong             |  6,808            |  20,090                   |

We report the frames per second (FPS) for a single instance of each of our environments below. With `multiprocessing`, environment FPS scales roughly linearly with the number of processes. Feel free to rerun this benchmark using [this colab notebook](https://colab.research.google.com/drive/1_ew-Piq5d9R_NkmP1lSzFX1fbK-swuAN?usp=sharing).

</p>
</details>

### Environment Descriptions

<details><summary>Expand</summary>
<p>

#### Concentration
  The quintessential memory game, sometimes known as "memory". A deck of cards is shuffled and placed face-down. The agent picks two cards to flip face up, if the cards match ranks, the cards are removed from play and the agent receives a reward. If they don't match, they are placed back face-down. The agent must remember where it has seen cards in the past.
#### Higher Lower
Guess whether the next card drawn from the deck is higher or lower than the previously drawn card. The agent should keep a count like blackjack and modify bets, but this game is significantly simpler than blackjack.
#### Battleship
One-player battleship. Select a gridsquare to launch an attack, and receive confirmation whether you hit the target. The agent should use memory to remember which gridsquares were hits and which were misses, completing an episode sooner.
#### Multiarmed Bandit
Over an episode, solve a multiarmed bandit problem by maximizing the expected reward. The agent should use memory to keep a running mean and variance of bandits.
#### Minesweeper
Classic minesweeper, but with reduced vision range. The agent only has vision of the surroundings near its last sweep. The agent must use memory to remember where the bombs are
#### Repeat Previous
Output the t-k<sup>th</sup> observation for a reward
#### Repeat First
Output the zeroth observation for a reward
#### Autoencode
The agent will receive k observations then must output them in the same order
#### Stateless Cartpole
Classic cartpole, except the velocity and angular velocity magnitudes are hidden. The agent must use memory to compute rates of change.
#### Noisy Stateless Cartpole
Stateless Cartpole with added Gaussian noise
#### Stateless Pendulum
Classic pendulum, but the velocity and angular velocity are hidden from the agent. The agent must use memory to compute rates of change.
#### Noisy Stateless Pendulum
Stateless Pendulum with added Gaussian noise
#### Labyrinth Escape
Escape randomly-generated labyrinths. The agent must remember wrong turns it has taken to find the exit.
#### Labyrinth Explore
Explore as much of the labyrinth as possible in the time given. The agent must remember where it has been to maximize reward.
#### Count Recall
The player is given a sequence of cards and is asked to recall how many times it has seen a specific card.

</p>
</details>

## POPGym Baselines
POPGym baselines implements recurrent and memory model in an efficient manner. POPGym baselines is implemented on top of [`rllib`](https://github.com/ray-project/ray) using their custom model API.

### Setup

<details><summary>Expand</summary>
<p>

To install the baselines and dependencies, first install ray
```bash
pip install "ray[rllib]==2.0.0"
```
`ray` must be installed separately, as it erroneously pins an old verison of gym and will cause dependency issues. Once ray is installed, install popgym either via pip or from source.

#### Pip 
```bash
pip install "popgym[baselines]"
```

### From Source
```bash
git clone https://github.com/smorad/popgym
cd popgym
pip install ".[baselines]"
```

</p>
</details>

### Usage

<details><summary>Expand</summary>
<p>


Our baselines exist in the `ray_models` directory. Here is how to use
the `GRU` model with `rllib`.
```python
import popgym
import ray
from torch import nn
from popgym.baselines.ray_models.ray_gru import GRU
# See what GRU-specific hyperparameters we can set
print(GRU.MODEL_CONFIG)
# Show other settable model hyperparameters like 
# what the actor/critic branches look like,
# what hidden size to use, 
# whether to add a positional embedding, etc.
print(GRU.BASE_CONFIG)
# How long the temporal window for backprop is
# This doesn't need to be longer than 1024
bptt_size = 1024
config = {
   "model": {
      "max_seq_len": bptt_size,
      "custom_model": GRU,
      "custom_model_config": {
        # Override the hidden_size from BASE_CONFIG
        # The input and output sizes of the MLP feeding the memory model
        "preprocessor_input_size": 128,
        "preprocessor_output_size": 64,
        "preprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
        # this is the size of the recurrent state in most cases
        "hidden_size": 128,
        # We should also change other parts of the architecture to use
        # this new hidden size
        # For the GRU, the output is of size hidden_size
        "postprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
        "postprocessor_output_size": 64,
        # Actor and critic networks
        "actor": nn.Linear(64, 64),
        "critic": nn.Linear(64, 64),
        # We can also override GRU-specific hyperparams
        "num_recurrent_layers": 1,
      },
   },
   # Some other rllib defaults you might want to change
   # See https://docs.ray.io/en/latest/rllib/rllib-training.html#common-parameters
   # for a full list of rllib settings
   # 
   # These should be a factor of bptt_size
   "sgd_minibatch_size": bptt_size * 4,
   # Should be a factor of sgd_minibatch_size
   "train_batch_size": bptt_size * 8,
   # The environment we are training on
   "env": "popgym-ConcentrationEasy-v0",
   # You probably don't want to change these values
   "rollout_fragment_length": bptt_size,
   "framework": "torch",
   "horizon": bptt_size,
   "batch_mode": "complete_episodes",
}
# Stop after 50k environment steps
ray.tune.run("PPO", config=config, stop={"timesteps_total": 50_000})
```

To add your own custom model, inherit from [BaseModel](popgym/baselines/ray_models/base_model.py) and implement the `initial_state` and `memory_forward` functions, as well as define your model configuration using `MODEL_CONFIG`. To use any of these or your own custom model in `ray`, make it the `custom_model` in the `rllib` config.

</p>
</details>


### Available Baselines

<details><summary>Expand</summary>
<p>

1. [MLP](popgym/baselines/ray_models/ray_mlp.py)
2. [Positional MLP](popgym/baselines/ray_models/ray_mlp.py)
3. [Framestacking](popgym/baselines/ray_models/ray_framestack.py) [(Paper)](https://arxiv.org/abs/1312.5602)
4. [Temporal Convolution Networks](popgym/baselines/ray_models/ray_frameconv.py) [(Paper)](https://arxiv.org/pdf/1803.01271.pdf)
5. [Elman Networks](https://github.com/smorad/popgym/blob/master/popgym/baselines/ray_models/ray_elman.py) [(Paper)](http://faculty.otterbein.edu/dstucki/COMP4230/FindingStructureInTime.pdf)
6. [Long Short-Term Memory](popgym/baselines/ray_models/ray_lstm.py) [(Paper)](http://www.bioinf.jku.at/publications/older/2604.pdf)
7. [Gated Recurrent Units](popgym/baselines/ray_models/ray_gru.py) [(Paper)](https://arxiv.org/abs/1412.3555)
8. [Independently Recurrent Neural Networks](popgym/baselines/ray_models/ray_indrnn.py) [(Paper)](https://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Li_Independently_Recurrent_Neural_CVPR_2018_paper.pdf)
9. [Fast Autoregressive Transformers](popgym/baselines/ray_models/ray_linear_attention.py) [(Paper)](https://proceedings.mlr.press/v119/katharopoulos20a.html)
10. [Fast Weight Programmers](popgym/baselines/ray_models/ray_fwp.py) [(Paper)](https://proceedings.mlr.press/v139/schlag21a.html) 
12. [Legendre Memory Units](popgym/baselines/ray_models/ray_lmu.py) [(Paper)](https://proceedings.neurips.cc/paper/2019/hash/952285b9b7e7a1be5aa7849f32ffff05-Abstract.html)
12. [Diagonal State Space Models](popgym/baselines/ray_models/ray_s4d.py) [(Paper)](https://arxiv.org/abs/2206.11893)
13. [Differentiable Neural Computers](popgym/baselines/ray_models/ray_diffnc.py) [(Paper)](http://clgiles.ist.psu.edu/IST597/materials/slides/papers-memory/2016-graves.pdf)

</p>
</details>


# Leaderboard

<details><summary>Expand</summary>
<p>


We provide a leaderboard of the best module in each environment. Using `ppo.py`, we run 3 trials of each trial. We compute the mean episodic reward over each batch, and store the maximum for each episode. We report the mean and standard deviations over the maximums, taken from at least 3 distinct trials.

The leaderboard is hosted at [paperswithcode](https://paperswithcode.com/dataset/popgym).

</p>
</details>

# Contributing

<details><summary>Expand</summary>
<p>


Steps to follow:
1. Fork this repo in github
2. Clone your fork to your machine
3. Move your environment into the forked repo
4. Install precommit in the fork (see below)
5. Write a unittest in `tests/`, see other tests for examples
6. Add your environment to `ALL_ENVS` in `popgym/__init__.py`
7. Make sure you don't break any tests by running `pytest tests/`
8. Git commit and push to your fork
9. Open a pull request on github


```bash
# Step 4. Install pre-commit in the fork
pip install pre-commit
git clone https://github.com/smorad/popgym
cd popgym
pre-commit install
```

</p>
</details>

# Citing
```
@inproceedings{
morad2023popgym,
title={{POPG}ym: Benchmarking Partially Observable Reinforcement Learning},
author={Steven Morad and Ryan Kortvelesy and Matteo Bettini and Stephan Liwicki and Amanda Prorok},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=chDrutUTs0K}
}
```
