import copy

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from popgym.baselines.ray_models.ray_diffnc import DiffNC
from popgym.baselines.ray_models.ray_elman import Elman
from popgym.baselines.ray_models.ray_frameconv import Frameconv
from popgym.baselines.ray_models.ray_framestack import Framestack
from popgym.baselines.ray_models.ray_fwp import FastWeightProgrammer
from popgym.baselines.ray_models.ray_gru import GRU
from popgym.baselines.ray_models.ray_indrnn import IndRNN
from popgym.baselines.ray_models.ray_linear_attention import LinearAttention
from popgym.baselines.ray_models.ray_lmu import LMU
from popgym.baselines.ray_models.ray_lstm import LSTM
from popgym.baselines.ray_models.ray_mlp import MLP, BasicMLP
from popgym.baselines.ray_models.ray_s4d import S4D

BATCH = 64
TIME = 1024
DIM = 256
h = 128
SAMPLES = 10


def main():
    cfg = {
        "max_seq_len": TIME,
        "custom_model_config": {
            "hidden_size": DIM,
            "preprocessor_input_size": h,
            "preprocessor": torch.nn.Identity(),
            "preprocessor_output_size": h,
            "postprocessor": torch.nn.Identity(),
            "actor": torch.nn.Identity(),
            "critic": torch.nn.Identity(),
            "postprocessor_output_size": DIM,
        },
    }
    obs_shape = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32)
    act_shape = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
    args = [obs_shape, act_shape, 1, cfg, "name"]
    rnn_cfg = copy.deepcopy(cfg)
    rnn_cfg["custom_model_config"]["benchmark"] = True
    rnn_args = [obs_shape, act_shape, 1, rnn_cfg, "name"]

    models = {
        "LSTM": [LSTM, rnn_args],
        "GRU": [GRU, rnn_args],
        "Elman": [Elman, rnn_args],
        "IndRNN": [IndRNN, args],
        "LMU": [LMU, args],
        "FART": [LinearAttention, args],
        "FWP": [FastWeightProgrammer, args],
        "S4D": [S4D, args],
        "MLP": [BasicMLP, args],
        "PosMLP": [MLP, args],
        "TCN": [Frameconv, args],
        "Fr.Stack": [Framestack, args],
        "DNC": [DiffNC, args],
    }
    results = []
    for name, (model, args) in models.items():
        print("TRAINING", name)
        results += train_closure(model(*args), "cuda")
        # print("# Params:", num_params)

    for name, (model, args) in models.items():
        print("EVALUATING", name)
        results += inference_closure(model(*args), "cpu")
        results += inference_closure(model(*args), "cuda")

    df = pd.DataFrame(results).sort_values(["mode", "device", "model"])
    df.to_csv("throughput.csv")
    # df.style.format(precision=2).hide_index().to_latex()
    breakpoint()


def train_closure(model, device):
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters())
    data = [torch.rand((BATCH, TIME, h), device=device) for i in range(SAMPLES)]
    seq_lens = torch.full((BATCH,), TIME, device=device)
    state = model.get_initial_state()
    state = [s.unsqueeze(0).repeat(BATCH, *([1] * s.dim())).to(device) for s in state]

    # Warm up kernels
    for i in range(2):
        _, _ = model.forward(
            {"obs_flat": torch.rand_like(data[0].reshape(BATCH * TIME, -1))},
            state,
            seq_lens,
        )
    del _
    opt.zero_grad()
    torch.cuda.synchronize()
    base_mem = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    num_params = sum([p.numel() for p in list(model.parameters()) if p.requires_grad])

    import time

    results = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(SAMPLES):
        start = time.time()
        opt.zero_grad()
        out, _ = model.forward(
            {"obs_flat": data[i].reshape(BATCH * TIME, -1)}, state, seq_lens
        )
        loss = out.mean()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        stop = time.time()
        results.append(
            {
                "model": model.__class__.__name__,
                "time": stop - start,
                "device": device,
                "mode": "train",
                "num_params": num_params,
            }
        )
    mem = torch.cuda.memory_stats()["allocated_bytes.all.peak"] - base_mem
    for r in results:
        r.update({"mem": mem})
    return results


def inference_closure(model, device):
    model = model.to(device)
    model.eval()
    data = [torch.rand((BATCH, 1, h), device=device) for i in range(SAMPLES)]
    seq_lens = torch.full((BATCH,), 1, device=device)
    state = model.get_initial_state()
    state = [s.unsqueeze(0).repeat(BATCH, *([1] * s.dim())).to(device) for s in state]

    # Warm up kernels
    with torch.no_grad():
        for i in range(2):
            _, _ = model.forward(
                {"obs_flat": torch.rand_like(data[0].reshape(BATCH, -1))},
                state,
                seq_lens,
            )
            del _
    if device != "cpu":
        torch.cuda.synchronize()

    import time

    num_params = sum([p.numel() for p in list(model.parameters()) if p.requires_grad])
    results = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(SAMPLES):
            start = time.time()
            for d in data:
                out, state = model.forward(
                    {"obs_flat": data[i].reshape(BATCH, -1)}, state, seq_lens
                )
                if device == "cuda":
                    torch.cuda.synchronize()
            stop = time.time()
            results.append(
                {
                    "model": model.__class__.__name__,
                    "time": (stop - start) / SAMPLES,
                    "device": device,
                    "mode": "inference",
                    "num_params": num_params,
                }
            )
    for r in results:
        r.update({"mem": 0})

    return results

    # print(
    #    f"{model.__class__.__name__} {device} sequence inference time:"
    #    f" {mu:.1f}+/-{std:.1f}ms memory: {mem / (1e6):.2f}MB"
    # )


if __name__ == "__main__":
    main()
