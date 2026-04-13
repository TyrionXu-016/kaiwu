# Run train_test inside the Kaiwu kaiwudrl container.
# Requires: Docker dev stack running (container name contains "kaiwudrl").
# Why not "cd /workspace/code && python3 train_test.py": kaiwudrl needs cwd at project root
# for configure.toml / tools/, and PYTHONPATH must include both /workspace/code (your agent) and project root.
#
# Usage from repo root:
#   powershell -ExecutionPolicy Bypass -File .\run_train_in_docker.ps1

$ErrorActionPreference = "Stop"

$ids = @(docker ps -q --filter "name=kaiwudrl" --filter "status=running")
if ($ids.Count -eq 0) {
    Write-Host "No running container matching name 'kaiwudrl'. Start the dev stack from dev\ (docker compose) first." -ForegroundColor Red
    exit 1
}

$cid = $ids[0]
Write-Host "Container: $cid" -ForegroundColor Cyan
Write-Host "  cd /data/projects/robot_vacuum" -ForegroundColor DarkGray
Write-Host "  PYTHONPATH=/workspace/code:/data/projects/robot_vacuum" -ForegroundColor DarkGray
Write-Host "  python3 /workspace/code/train_test.py" -ForegroundColor DarkGray

docker exec $cid bash -lc "cd /data/projects/robot_vacuum && export PYTHONPATH=/workspace/code:/data/projects/robot_vacuum && python3 /workspace/code/train_test.py"
