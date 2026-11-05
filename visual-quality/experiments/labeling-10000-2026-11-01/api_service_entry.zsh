#!/usr/bin/zsh
set -eu
source ~/.config/zsh/.zsh-private >/dev/null 2>&1
exec /home/ahstn/git/model-gym/visual-quality/.venv/bin/python \
  /tmp/model-gym-10000-20261101/visual-quality/experiments/labeling-10000-2026-11-01/api_runner.py "$@"
