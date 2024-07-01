"""Car Flag tasks a car with driving across a 1D line to the correct flag. 

The car must first drive to the oracle flag and then to the correct endpoint. 
The agent's observation is a vector of three floats: its position on the line, 
its velocity at each timestep, and the goal flag's location when it reaches 
the oracle flag. The agent's actions alter its velocity: it can accelerate left, 
perform a no-op (maintain current velocity), or accelerate right."""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from popgym.core.env import POPGymEnv


class CarFlag(POPGymEnv):
    """Car Flag tasks a car with driving across a 1D line to the correct flag.

    The car must first drive to the oracle flag and then to the correct endpoint.
    The agent's observation is a vector of three floats: its position on the line,
    its velocity at each timestep, and the goal flag's location when it reaches
    the oracle flag. The agent's actions alter its velocity: it can accelerate left,
    perform a no-op (maintain current velocity), or accelerate right.

    Args:
        discrete: True, or False. Sets the action space to discrete or continuous.

    Returns:
        A gym environment
    """

    def __init__(self, discrete=True, difficulty="easy"):
        assert difficulty in ["easy", "medium", "hard"]
        if difficulty == "easy":
            self.heaven_position = 1.0
            self.hell_position = -1.0
        elif difficulty == "medium":
            self.heaven_position = 3.0
            self.hell_position = -3.0
        elif difficulty == "hard":
            self.heaven_position = 5.0
            self.hell_position = -5.0
        else:
            raise NotImplementedError(f"Invalid difficulty {difficulty}")
        self.max_position = self.heaven_position + 0.1
        self.min_position = -self.max_position
        self.max_speed = 0.07

        self.min_action = -1.0
        self.max_action = 1.0

        self.heaven_position = 1.0
        self.hell_position = -1.0
        self.oracle_position = 0.5
        self.power = 0.0015

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        # When the cart is within this vicinity, it observes the direction given
        # by the oracle
        self.oracle_delta = 0.2

        self.low_state = np.array(
            [self.min_position, -self.max_speed, -1.0], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed, 1.0], dtype=np.float32
        )

        self.discrete = discrete

        if self.discrete:
            self.action_space = gym.spaces.Discrete(3)
        else:
            self.action_space = gym.spaces.Box(
                low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Box(
            low=self.low_state, high=self.high_state, shape=(3,), dtype=np.float32
        )

        self.np_random = None
        self.state = None

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        if self.discrete:
            # 0 is -1, 1 is 0, 2 is 1
            force = action - 1
        else:
            force = np.clip(action, -1, 1)

        velocity += force * self.power
        velocity = min(velocity, self.max_speed)
        velocity = max(velocity, -self.max_speed)
        position += velocity
        position = min(position, self.max_position)
        position = max(position, self.min_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        max_position = max(self.heaven_position, self.hell_position)
        min_position = min(self.heaven_position, self.hell_position)

        done = bool(position >= max_position or position <= min_position)

        env_reward = 0

        if self.heaven_position > self.hell_position:
            if position >= self.heaven_position:
                env_reward = 1.0

            if position <= self.hell_position:
                env_reward = -1.0

        if self.heaven_position < self.hell_position:
            if position <= self.heaven_position:
                env_reward = 1.0

            if position >= self.hell_position:
                env_reward = -1.0

        direction = 0.0
        if (
            position >= self.oracle_position - self.oracle_delta
            and position <= self.oracle_position + self.oracle_delta
        ):
            if self.heaven_position > self.hell_position:
                # Heaven on the right
                direction = 1.0
            else:
                # Heaven on the left
                direction = -1.0

        self.state = np.array([position, velocity, direction])

        return self.state, env_reward, done, {"is_success": env_reward > 0}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[gym.core.ObsType, Dict[str, Any]]:
        super().reset(seed=seed)
        # Randomize the heaven/hell location
        if self.np_random.integers(low=0, high=2, size=1) == 0:
            self.heaven_position = 1.0
        else:
            self.heaven_position = -1.0

        self.hell_position = -self.heaven_position

        self.state = np.array([self.np_random.uniform(low=-0.2, high=0.2), 0, 0.0])
        return np.array(self.state), {}

    def get_state(self):
        # Return the position of the car, oracle, and goal
        return (
            self.state,
            self.oracle_position,
            self.heaven_position,
            self.hell_position,
        )

    def render(self):
        return None


if __name__ == "__main__":
    e = CarFlag()
    obs = e.reset()
    e.render()
    while not done:
        action = np.array(input("Enter action: ")).astype(np.int8)
        obs, reward, done, info = e.step(action)
        print(f"reward = {reward}")


class CarFlagEasy(CarFlag):
    """Car Flag tasks a car with driving across a 1D line to the correct flag.
    The easy level has the range [-1, 1]."""

    def __init__(self):
        super().__init__("easy")


class CarFlagMedium(CarFlag):
    """Car Flag tasks a car with driving across a 1D line to the correct flag.
    The medium level has the range [-3, 3]."""

    def __init__(self):
        super().__init__("medium")


class CarFlagHard(CarFlag):
    """Car Flag tasks a car with driving across a 1D line to the correct flag.
    The hard level has the range [-5, 5]."""

    def __init__(self):
        super().__init__("hard")
