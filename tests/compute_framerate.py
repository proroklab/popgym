import multiprocessing
import time

import popgym

NUM_STEPS = 100000
NUM_WORKERS = 2


def run_sample(e, num_steps):
    env = e()
    env.reset()
    start = time.time()
    for i in range(NUM_STEPS):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    end = time.time()
    elapsed = end - start
    fps = NUM_STEPS / elapsed
    return fps


def main():
    for e in popgym.envs.ALL_BASE:
        fps = run_sample(e, NUM_STEPS)
        print(f"{e.__name__} (1x) FPS: {fps:.0f}")

    p = multiprocessing.Pool(processes=NUM_WORKERS)
    with p:
        for e in popgym.envs.ALL_BASE:
            envs = NUM_WORKERS * [e]
            steps = NUM_WORKERS * [int(NUM_STEPS / NUM_WORKERS)]
            fps = sum(p.starmap(run_sample, zip(envs, steps)))
            print(f"{e.__name__} ({NUM_WORKERS}x) FPS: {fps:.0f}")


if __name__ == "__main__":
    main()
