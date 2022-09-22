import numpy as np

from popgym.envs.count_recall import CountRecall

if __name__ == "__main__":
    game = CountRecall()
    done = False
    obs, info = game.reset(return_info=True)
    reward = -float("inf")
    game.render()

    while not done:
        action = input("Input action: ")
        obs, reward, done, info = game.step(np.array([action], dtype=np.float32))
        game.render()
        print("Received reward:", reward)
