from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import re


def get_overcooked_dir():
    return os.path.dirname(os.path.abspath(__file__))


def get_marl_grid_dir():
    return os.path.dirname(get_overcooked_dir())


def get_repo_root():
    return os.path.dirname(get_marl_grid_dir())


def get_workspace_root():
    return os.path.dirname(get_repo_root())


def get_find_goal_dir():
    return os.path.join(get_marl_grid_dir(), "find-goal")


def get_overcooked_venv_python():
    return os.path.join(
        get_workspace_root(), "overcooked_ai-master", ".venv", "bin", "python"
    )


def _same_interpreter(path_a, path_b):
    return os.path.realpath(path_a) == os.path.realpath(path_b)


def maybe_reexec_into_overcooked_venv():
    venv_python = get_overcooked_venv_python()
    if not os.path.exists(venv_python):
        return

    if _same_interpreter(sys.executable, venv_python):
        return

    # Overcooked dependencies are pinned to Python 3.10 inside its own venv.
    os.execv(venv_python, [venv_python] + sys.argv)


def path_matches_current_python(site_packages_path):
    match = re.search(r"python(\d+)\.(\d+)", site_packages_path)
    if match is None:
        return False
    major, minor = int(match.group(1)), int(match.group(2))
    return (major, minor) == sys.version_info[:2]


def ensure_shared_training_path():
    marl_grid_dir = get_marl_grid_dir()
    if marl_grid_dir not in sys.path:
        sys.path.insert(0, marl_grid_dir)
    find_goal_dir = get_find_goal_dir()
    if find_goal_dir not in sys.path:
        sys.path.insert(0, find_goal_dir)
    return find_goal_dir
