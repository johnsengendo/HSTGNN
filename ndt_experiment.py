"""Thin entrypoint for the HSTGNN Network Digital Twin experiment.

The implementation now lives in `ndt_project.pipeline` so the repository has a
cleaner separation between the script users run and the experiment logic.
"""

from ndt_project.pipeline import main


if __name__ == "__main__":
    main()
