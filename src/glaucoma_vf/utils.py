from pathlib import Path

import git
import os


def get_git_root(path) -> Path:
    # Check if we explicitly defined the root (for running in Containers)
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root)

    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")

    return Path(git_root)
