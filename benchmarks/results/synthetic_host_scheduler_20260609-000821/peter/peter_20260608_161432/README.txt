Synthetic host scheduler benchmark on peter: BLOCKED before smoke.

Remote result dir: /home/ubuntu/vllm_synthetic_host_benchmark/peter_20260608_161432
Local copied dir: /home/zyz/vLLM_deploy/vllm/benchmarks/results/_incoming_peter/peter_20260608_161432
Timestamp: 20260608_161432
Remote repo inspected: /home/ubuntu/vllm_p550dev_validation using /home/ubuntu/vllm/.venv-p550
Existing deployment repo left untouched: /home/ubuntu/vllm

Blocker:
The vLLM CLI needed by vllm bench synthetic fails before registering/running benchmark subcommands because the Python package model_hosting_container_standards is missing.

Exact reproducer:
cd /home/ubuntu/vllm_p550dev_validation
source /home/ubuntu/vllm/.venv-p550/bin/activate
python -m vllm.entrypoints.cli.main bench --help

Likely unblock command, if dependency installation is approved:
python -m pip install model-hosting-container-standards
