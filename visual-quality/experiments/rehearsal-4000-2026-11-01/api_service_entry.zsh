#!/usr/bin/env zsh
# Credential values are loaded in the service process, never in the unit definition.
set +x
source /home/ahstn/.config/zsh/.zsh-private >/dev/null 2>&1
export RUN_INFRA_API_KEY OPENROUTER_API_KEY
cd /tmp/model-gym-rehearsal-20261101 || exit 1
exec /home/ahstn/git/model-gym/visual-quality/.venv/bin/python -u "$@"
