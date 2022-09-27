from popgym.envs.minesweeper import MineSweeperHard

if __name__ == "__main__":
    game = MineSweeperHard()
    done = False
    obs, info = game.reset(return_info=True)
    reward = -float("inf")
    game.render()

    while not done:
        action = input("input index:").split(',')
        action = (int(action[0]), int(action[1]))
        obs, reward, done, info = game.step(action)
        game.render()
        print("reward:", reward)
