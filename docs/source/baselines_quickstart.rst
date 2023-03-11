Baselines Quickstart
----------------------

The baselines contain 13 different memory models, built to work with RLlib. In the following example, we will show you how to run a GRU but with different hidden and recurrent sizes than the original paper. See the ray_models directory for other models.

.. code-block:: python

    import popgym
    import ray
    from torch import nn
    from popgym.baselines.ray_models.ray_gru import GRU
    # See what GRU-specific hyperparameters we can set
    print(GRU.MODEL_CONFIG)
    # Show other settable model hyperparameters like
    # what the actor/critic branches look like,
    # what hidden size to use,
    # whether to add a positional embedding, etc.
    print(GRU.BASE_CONFIG)
    # How long the temporal window for backprop is
    # This doesn't need to be longer than 1024
    bptt_size = 1024
    config = {
    "model": {
        "max_seq_len": bptt_size,
        "custom_model": GRU,
        "custom_model_config": {
        # Override the hidden_size from BASE_CONFIG
        # The input and output sizes of the MLP feeding the memory model
        "preprocessor_input_size": 128,
        "preprocessor_output_size": 64,
        "preprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
        # this is the size of the recurrent state in most cases
        "hidden_size": 128,
        # We should also change other parts of the architecture to use
        # this new hidden size
        # For the GRU, the output is of size hidden_size
        "postprocessor": nn.Sequential(nn.Linear(128, 64), nn.ReLU()),
        "postprocessor_output_size": 64,
        # Actor and critic networks
        "actor": nn.Linear(64, 64),
        "critic": nn.Linear(64, 64),
        # We can also override GRU-specific hyperparams
        "num_recurrent_layers": 1,
        },
    },
    # Some other rllib defaults you might want to change
    # See https://docs.ray.io/en/latest/rllib/rllib-training.html#common-parameters
    # for a full list of rllib settings
    #
    # These should be a factor of bptt_size
    "sgd_minibatch_size": bptt_size * 4,
    # Should be a factor of sgd_minibatch_size
    "train_batch_size": bptt_size * 8,
    # The environment we are training on
    "env": "popgym-ConcentrationEasy-v0",
    # You probably don't want to change these values
    "rollout_fragment_length": bptt_size,
    "framework": "torch",
    "horizon": bptt_size,
    "batch_mode": "complete_episodes",
    }
    # Stop after 50k environment steps
    ray.tune.run("PPO", config=config, stop={"timesteps_total": 50_000})

To add your own custom memory model, inherit from 
:py:mod:`popgym.baselines.ray_models.base_model` and implement the ``initial_state`` and ``memory_forward`` functions, 
as well as define your model configuration using ``MODEL_CONFIG``. To use any of these or your own custom model 
in ray, make it the custom_model in the rllib config.
