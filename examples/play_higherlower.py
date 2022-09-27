from popgym.envs.higher_lower import HigherLower

if __name__ == "__main__":
    game = HigherLower()
    done = False
    obs, info = game.reset(return_info=True)
    reward = -float("inf")
    game.render()

    while not done:
        action = input("input index:")
        obs, reward, done, info = game.step(int(action))
        print(game.render())
        print("reward:", reward)
