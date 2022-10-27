# Partially Observable Process Gym (POPGym)
![tests](https://github.com/smorad/popgym/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/smorad/popgym/branch/master/graph/badge.svg?token=I47IDFZXSV)](https://codecov.io/gh/smorad/popgym)

POPGym is designed to benchmark memory in deep reinforcement learning. It contains a set of [environments](#popgym-environments) and a collection of [memory model baselines](#popgym-baselines).

## Setup
Packages will be sent to pypi soon. Until then, to install the environments:
```bash
git clone https://github.com/smorad/popgym
cd popgym
pip install .
```

To install the baselines and dependencies, first install ray
```bash
pip install ray[rllib]
```
You must do this, as ray-2.0.0 erroneously pins an old verison of gym and will cause dependency issues. This has been patched but did not make it into the latest release. Once ray is installed, install popgym:
```bash
git clone https://github.com/smorad/popgym
cd popgym
pip install ".[baselines]"
```

## POPGym Environments

POPGym contains Partially Observable Markov Decision Process (POMDP) environments following the [Openai Gym](https://github.com/openai/gym) interface, where every single environment is procedurally generated. We find that much of RL is a huge pain-in-the-rear to get up and running, so our environments follow a few basic tenets:

1. **Painless setup** - `popgym` environments requires only `gym`, `numpy`, and `mazelib` as core dependencies, and can be installed with a single `pip install`.
2. **Laptop-sized tasks** - None of our environments have large observation spaces or require GPUs to render. Well-designed models should be able to solve a majority of tasks in less than a day.
3. **True generalization** - It is possible for memoryless agents to receive high rewards on environments by memorizing the layout of each level. To avoid this, all environments are heavily randomized. 

### Environment Overview
The environments are split into set or sequence tasks. Ordering matters in sequence tasks (e.g. the order of the button presses in simon matters), and does not matter in set tasks (e.g. the "count" in blackjack does not change if you swap o<sub>t-1</sub> and o<sub>t-k</sub>). We provide a table of the environments. The frames per second (FPS) was computed by running the `popgym-perf-test.ipynb` notebook on the Google Colab free tier by stepping and resetting single environment for 100k timesteps. We also provide the same benchmark run on a Macbook Air (2020). With `multiprocessing`, environment FPS scales roughly linearly with the number of processes.

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

Feel free to rerun this benchmark using [this colab notebook](https://colab.research.google.com/drive/1_ew-Piq5d9R_NkmP1lSzFX1fbK-swuAN?usp=sharing).


### Environment Descriptions
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


## POPGym Baselines
We implement the following baselines as `RLlib` custom models:

1. [MLP](popgym/baselines/ray_models/ray_mlp.py)
2. [Positional MLP](popgym/baselines/ray_models/ray_mlp.py)
3. [Framestacking](popgym/baselines/ray_models/ray_framestack.py)
4. [Temporal Convolution](popgym/baselines/ray_models/ray_frameconv.py)
5. [Elman Networks](https://github.com/smorad/popgym/blob/master/popgym/baselines/ray_models/ray_elman.py)
6. [Long Short-Term Memory](popgym/baselines/ray_models/ray_lstm.py)
7. [Gated Recurrent Units](popgym/baselines/ray_models/ray_gru.py)
8. [Independently Recurrent Neural Networks](popgym/baselines/ray_models/ray_indrnn.py)
9. [Fast Autoregressive Transformers](popgym/baselines/ray_models/ray_linear_attention.py)
10. [Fast Weight Programmers](popgym/baselines/ray_models/ray_fwp.py)
12. [Legendre Memory Units](popgym/baselines/ray_models/ray_lmu.py)
12. [Diagonal State Space Models](popgym/baselines/ray_models/ray_s4d.py)
13. [Differentiable Neural Computers](popgym/baselines/ray_models/ray_diffnc.py)

To add your own custom model, please inherit from [BaseModel](popgym/baselines/ray_models/base_model.py) and implement the `initial_state` and `memory_forward` functions, as well as define your model configuration using `MODEL_CONFIG`. To use any of these or your own custom model in `ray`, simply add it to the `ray` config:

```python
import ray
from popgym.baselines.ray_models.ray_lstm import LSTM
config = {
   ...
   "model": {
      custom_model: LSTM,
      custom_model_config: {"hidden_size": 128}
   }
}
ray.tune.run("PPO", config)
```
Each model defines a MODEL_CONFIG that you can set by adding keys and values to `custom_model_config`. See [ppo.py](popgym/baselines/ppo.py) for an in-depth example.


## Contributing
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

## Citing
If you found POPGym useful, please cite it
```bibtex
@misc{
  morad_kortvelesy_prorok_2022, 
  url={https://github.com/smorad/popgym}, 
  journal={POPGym: Partially Observable Process Gym}, 
  publisher={GitHub}, 
  author={Morad, Steven and Kortvelesy, Ryan and Prorok, Amanda}, 
  year={2022}, 
  month={Sep}
}
```
