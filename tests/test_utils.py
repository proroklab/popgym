import warnings
from gymnasium.utils.env_checker import check_env


RAW_ENV_CHECK_WARNING = (
    r".*The environment \(<.*>\) is different from the unwrapped "
    r"version \(<.*>\)\. This could effect the environment checker as the "
    r"environment most likely has a wrapper applied to it\. We recommend "
    r"using the raw environment for `check_env` using `env\.unwrapped`\."
)
BOX_OBS_MAX_INFINITY_WARNING = (
    r".*A Box observation space .* value is"
)

BOX_ACTION_WARNING = r".*we recommend using a symmetric and normalized space.*"


def check_env_no_warnings(env):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=RAW_ENV_CHECK_WARNING,
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=BOX_OBS_MAX_INFINITY_WARNING,
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=BOX_ACTION_WARNING,
            category=UserWarning,
        )
        check_env(env, skip_render_check=True)