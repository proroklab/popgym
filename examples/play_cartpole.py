from popgym.envs.stateless_cartpole import PositionOnlyCartPole

if __name__ == "__main__":
    game = PositionOnlyCartPole()
    done = False
    obs, info = game.reset(return_info=True)
    reward = -float("inf")
    game.render()

    while not done:
        action = input("input index:")
        obs, reward, done, info = game.step(int(action))
        game.render()
        print("reward:", reward)
