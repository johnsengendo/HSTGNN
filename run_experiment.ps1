$ErrorActionPreference = "Stop"

if (Test-Path ".\.venv\Scripts\python.exe") {
    & .\.venv\Scripts\python.exe .\ndt_experiment.py --mode single --single-output-dir zoo_results
} else {
    python .\ndt_experiment.py --mode single --single-output-dir zoo_results
}
