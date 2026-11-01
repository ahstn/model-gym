#!/bin/zsh
set -e
set +x
source ~/.config/zsh/.zsh-private >/dev/null 2>&1
export OPENROUTER_API_KEY
p=/tmp/model-gym-rehearsal-20261101/visual-quality/experiments/eligibility-expanded-120-2026-11-01
/home/ahstn/git/model-gym/visual-quality/.venv/bin/python -c 'import hashlib,json,pathlib; p=pathlib.Path("/tmp/model-gym-rehearsal-20261101/visual-quality/experiments/eligibility-expanded-120-2026-11-01"); a=json.loads((p/"holdout-driver-amendment.json").read_text()); assert hashlib.sha256((p/"drive_holdout.py").read_bytes()).hexdigest()==a["corrected_driver_sha256"]; assert hashlib.sha256((p/"winner.json").read_bytes()).hexdigest()==a["winner_sha256"]'
exec /home/ahstn/git/model-gym/visual-quality/.venv/bin/python "$p/drive_holdout.py" holdout
