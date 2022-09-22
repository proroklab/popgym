import pygame

from popgym.envs.labyrinth_explore import LabyrinthExplore

if __name__ == "__main__":
    e = LabyrinthExplore()
    obs = e.reset(seed=0)
    e.render()
    done = False
    rewards = 0

    pygame.init()
    while not done:
        events = pygame.event.get()
        action = None
        # breakpoint()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 0
                if event.key == pygame.K_RIGHT:
                    action = 1
                if event.key == pygame.K_UP:
                    action = 2
                if event.key == pygame.K_DOWN:
                    action = 3
        if action is not None:
            obs, reward, done, info = e.step(action)
            e.render()
            print(reward)
            rewards += reward
    print("final reward", rewards)
