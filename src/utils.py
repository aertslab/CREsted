import yaml
import os


def load_config(
    default_path: str = "configs/default.yml",
    user_path: str = "configs/user.yml",
):
    """Load user configs. Create if don't exist. Copy missing configs from default."""
    # If user.yaml doesn't exist, create it from default.yaml
    if not os.path.exists(user_path):
        with open(default_path, "r") as df, open(user_path, "w") as uf:
            uf.write(df.read())
        return _open_config(default_path)

    # If user.yaml exists, merge with default.yaml
    default_config = _open_config(default_path)
    user_config = _open_config(user_path)

    # Add any missing keys from default to user config and save
    updated = False
    for key, value in default_config.items():
        if key not in user_config:
            user_config[key] = value
            updated = True

    if updated:
        with open(user_path, "w") as uf:
            yaml.safe_dump(user_config, uf)

    return user_config


def _open_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
