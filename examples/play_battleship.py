from popgym.envs.battleship import Battleship

if __name__ == "__main__":
    game = Battleship()
    done = False
    obs, info = game.reset(return_info=True)
    reward = -float("inf")
    game.render()

    while not done:
        action = input("input index:").split(",")
        obs, reward, done, info = game.step(((int(action[0]), int(action[1]))))
        game.render()
        print("reward:", reward)
