"""Shared config: target file list, repo path, model names."""
from pathlib import Path

TARGET_FILES = {
    "_api", "_client", "_models", "_config",
    "_auth", "_urls", "_exceptions", "_types", "_utils",
}

GIT_REPO_ROOT = Path("target_repos/httpx")
REPO_PATH = Path("target_repos/httpx/httpx")