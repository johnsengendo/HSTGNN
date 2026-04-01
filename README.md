# HSTGNN Network Digital Twin

This repository runs the HSTGNN experiment for Network Digital Twins and generates:

- `ndt_results.csv`
- `bar_r2_score.png`
- `bar_mae_rmse.png`
- `radar_chart.png`
- `training_curves.png`
- `scatter_pred_vs_actual.png`
- topology visualisations

## Files to Push

At minimum, push these files:

- `ndt_experiment.py`
- `ndt_project/pipeline.py`
- `requirements.txt`
- `run_experiment.ps1`
- `.gitignore`

If you also want the PDF summary workflow, include:

- `generate_zoo_pdf_report.py`

## Dataset Requirement

The script expects the Internet Topology Zoo dataset at:

`3D-internet-zoo-master/3D-internet-zoo-master/dataset`

If you keep that folder in the repository, the experiment can run directly.

## Quick Start

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
.\run_experiment.ps1
```

### Direct Commands

Single combined-topology run:

```powershell
python ndt_experiment.py --mode single --single-output-dir zoo_results
```

Multi-topology suite:

```powershell
python ndt_experiment.py --mode multi --output-root multi_topology_runs
```

Generate the PDF summary report after a `zoo_results` run:

```powershell
python generate_zoo_pdf_report.py
```

## Outputs

- `zoo_results/` is a good folder name for a single presentation-ready run.
- `multi_topology_runs/` stores per-topology benchmark outputs for multi-mode experiments.

## Notes

- `ndt_experiment.py` is now a thin entrypoint. The main experiment logic lives in `ndt_project/pipeline.py`.
- The experiment is deterministic with a fixed seed by default.
- The script automatically generates plots and CSV outputs at the end of each run.
- If CUDA is available, PyTorch will use it automatically; otherwise the run falls back to CPU.
