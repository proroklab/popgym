.. _environment-quickstart:

Environment Quickstart
----------------------

Let's create an environment and add some wrappers to it. First, let's do all required imports, then print all the available environments

.. code-block:: python

    import gymnasium as gym
    import popgym
    from popgym.wrappers import PreviousAction, Antialias, Markovian, Flatten, DiscreteAction
    from popgym.core.observability import Observability, STATE
    env_classes = popgym.envs.ALL.keys()
    print(env_classes)

You can also browse the full list of environments with proper descriptions at :py:mod:`popgym.envs`.
now let's create a stateless cartpole environment

.. code-block:: python

    env = popgym.envs.position_only_cartpole.PositionOnlyCartPoleEasy()

We also might want to add some wrappers. In POMDPs, we often condition on the previous action. We can do this using the PreviousAction wrapper.

.. code-block:: python

    wrapped_env = PreviousAction(env)

At the initial timestep, there is no previous action. By default, PreviousAction will return a zero action. To prevent aliasing for the initial, we can add an indicator to the observation space, indicating whether this is the initial timestep.

.. code-block:: python

    wrapped_env = Antialias(wrapped_env)

Many RL libraries have spotty support for nested observations or MultiDiscrete action spaces. If you are using DQN or similar approaches, you might want to flatten the observation and action spaces, then convert the action space into a single large Discrete space

.. code-block:: python

    DiscreteAction(Flatten(wrapped_env))

We will not actually assign this to wrapped env, as for this example we want to inspect the observation and action spaces. Finally, we can decide if we want the hidden Markov state. We can add it as part of the observation, into the info dict, etc. See Observability for more options.

.. code-block:: python

    wrapped_env = Markovian(wrapped_env, Observability.FULL_IN_INFO_DICT)

Now, let's run the environment and see what the observation looks like

.. code-block:: python

    wrapped_env.reset()
    obs, reward, terminated, truncated, info = wrapped_env.step(wrapped_env.action_space.sample())

This will return the linear and angular velocities, the previous action, and the antialias indicator.

.. code-block:: python

    print(obs)
    >>> (array([0.0348076 , 0.02231686], dtype=float32), 1, 0)

We can also print the underlying Markov state

.. code-block:: python

    print(info[STATE])
    >>> array([ 0.0348076 ,  0.14814377,  0.02231686, -0.31778395], dtype=float32)

If you are writing your own simple implementation, the flatten wrapper might be beneficial. It will flatten nested observation and action spaces into a single space.

.. code-block:: python

    wrapped_env = Flatten(wrapped_env)
    print(wrapped_env.action_space)
