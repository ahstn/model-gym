#!/bin/zsh
set -e
set +x
source ~/.config/zsh/.zsh-private >/dev/null 2>&1
export OPENROUTER_API_KEY
exec /home/ahstn/git/model-gym/visual-quality/.venv/bin/python /tmp/model-gym-rehearsal-20261101/visual-quality/experiments/eligibility-expanded-120-2026-11-01/drive.py "$1"
