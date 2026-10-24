# Run Month 1 experiments sequentially (PowerShell)
Set-Location $PSScriptRoot\..
$ErrorActionPreference = "Continue"

Write-Host "=== Week 1 (done manually or re-run) ==="
# python experiments/run_main_track.py --profile week1 --tasks regret,scm,pareto
# python experiments/make_main_track_figures.py

Write-Host "=== SCM-50 (overnight) ==="
python experiments/run_main_track.py --profile scm50 --tasks scm 2>&1 | Tee-Object results/main_track/scm50_run.log

Write-Host "=== IHDP + K-study ==="
python experiments/run_main_track.py --profile ihdp --tasks ihdp 2>&1 | Tee-Object results/main_track/ihdp_run.log
python experiments/run_main_track.py --profile kstudy --tasks kstudy 2>&1 | Tee-Object results/main_track/kstudy_run.log

python experiments/make_main_track_figures.py
Write-Host "Done. See results/main_track/MONTH1_PROGRESS.md"
