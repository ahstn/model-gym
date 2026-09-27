#!/usr/bin/env bash
# Stock (no adapter) dev-panel evals of candidate bases with the calib-fitted per-kind temperatures.
# Run on the pod from the decision/ directory. Weights for all but the cached base go to /workspace/hf (network
# volume) because the container disk is small. Logs: runs/logs/stock-<name>.log, exit code in .exit.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/logs

run() {
	local name=$1 model=$2 hf=$3
	shift 3
	HF_HOME=$hf uv run decision eval --model "$model" --panel data/panels/dev.jsonl --calib data/panels/calib.jsonl \
		--out "runs/stock-$name/eval" "$@" >"runs/logs/stock-$name.log" 2>&1
	echo $? >"runs/logs/stock-$name.exit"
}

run gemma-4-12b google/gemma-4-12B-it /root/hf
run qwen3.5-9b Qwen/Qwen3.5-9B /workspace/hf
run k2-horizon-7b IFM/K2-Horizon-7B /workspace/hf --trust-remote-code
run gemma-4-26b-a4b google/gemma-4-26B-A4B-it /workspace/hf --batch-tokens 16384
