from popgym.envs.blackjack import BlackJack, Phase

if __name__ == "__main__":
    game = BlackJack()
    done = False
    obs, info = game.reset(return_info=True)
    reward = -float("inf")
    game.render()
    phase: int = obs["phase"]
    action_dict = {"bet_size": 0, "hit": 0}

    while not done:
        if phase == Phase.BET:
            action = input(f"How much to bet? Input index: {game.bet_sizes} ")
            action_dict = {"bet_size": int(action), "hit": 0}
        elif phase == Phase.PLAY or phase == Phase.DEAL:
            action = input("Stay (0) or hit (1)?")
            action_dict = {"bet_size": 0, "hit": int(action)}
        elif phase == Phase.PAYOUT:
            action = input(f"Received reward of {reward}, any key to continue.")
        else:
            action = input(f"phase {phase}")

        obs, reward, done, info = game.step(action_dict)
        phase = obs["phase"]
        game.render()
        print(info)
