# POPGym: Partially Observable Process Gym
![tests](https://github.com/smorad/popgym/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/smorad/popgym/branch/master/graph/badge.svg?token=I47IDFZXSV)](https://codecov.io/gh/smorad/popgym)

POPGym is designed to benchmark memory in deep reinforcement learning. It contains a set of [environments](#popgym-environments) and a collection of [memory model baselines](#popgym-baselines). The full paper is available on [OpenReview](https://openreview.net/forum?id=chDrutUTs0K). Please see the [documentation](https://popgym.readthedocs.io/en/latest/) for advanced installation instructions and examples. 

## Quickstart Install

```python
pip install popgym # base environments only, only requires numpy and gymnasium
pip install "popgym[navigation]" # also include navigation environments, which require mazelib
pip install "popgym[baselines]" # environments and memory baselines
```

## POPGym Environments

POPGym contains Partially Observable Markov Decision Process (POMDP) environments following the [Gymnasium](https://github.com/Farama-Foundation/Gymnaisum) interface. POPGym environments have minimal dependencies and fast enough to solve on a laptop CPU in less than a day. We provide the following environments:

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

## POPGym Baselines
POPGym baselines implements recurrent and memory model in an efficient manner. POPGym baselines is implemented on top of [`rllib`](https://github.com/ray-project/ray) using their custom model API. We provide the following baselines:

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

# Leaderboard

The leaderboard is available at [paperswithcode](https://paperswithcode.com/dataset/popgym).

# Contributing
Follow style and ensure tests pass

```python
pip install pre-commit
pre-commit install
pytest popgym/tests
```

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
