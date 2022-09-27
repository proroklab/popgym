from popgym.envs.stateless_pendulum import StatelessPendulum

if __name__ == "__main__":
    game = StatelessPendulum()
    done = False
    obs, info = game.reset(return_info=True)
    reward = -float("inf")
    game.render()

    while not done:
        action = input("input index:")
        obs, reward, done, info = game.step(int(action))
        game.render()
        print("reward:", reward)
