#!/bin/zsh
set -eu
p=/tmp/model-gym-rehearsal-20261101/visual-quality/experiments/eligibility-expanded-120-2026-11-01
unit=modelgym-eligibility-expanded-dev-20261101.service
while [[ "$(systemctl --user show "$unit" -p ActiveState --value)" == "active" ]]; do sleep 5; done
[[ "$(systemctl --user show "$unit" -p Result --value)" == "success" ]]
[[ "$(systemctl --user show "$unit" -p ExecMainStatus --value)" == "0" ]]
[[ ! -e "$p/winner.json" ]]
/home/ahstn/git/model-gym/visual-quality/.venv/bin/python -c 'import hashlib,json,pathlib; p=pathlib.Path("/tmp/model-gym-rehearsal-20261101/visual-quality/experiments/eligibility-expanded-120-2026-11-01"); expected=json.loads((p/"selection-analysis-freeze.json").read_text())["sha256"]; assert hashlib.sha256((p/"analyze_expanded.py").read_bytes()).hexdigest()==expected'
/home/ahstn/git/model-gym/visual-quality/.venv/bin/python "$p/analyze_expanded.py" select
exec /bin/zsh "$p/run.zsh" holdout
