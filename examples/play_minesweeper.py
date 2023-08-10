from popgym.envs.minesweeper import MineSweeperEasy

if __name__ == "__main__":
    game = MineSweeperEasy()
    done = False
    obs, info = game.reset()
    reward = -float("inf")
    game.render()
    done = False

    while not done:
        action = input("input index:").split(",")
        action_int = (int(action[0]), int(action[1]))
        obs, reward, truncated, terminated, info = game.step(action_int)
        done = truncated or terminated
        print(obs)
        # game.render()
        print("reward:", reward)
