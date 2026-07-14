from __future__ import annotations


HELP = """SCOPE Artifact Evaluation bundle

Evidence-only (no GPU):
  make setup
  make evidence
  make validate

Run one experiment with Slurm (default):
  make fig-16
  make tbl-5
  make fig-17
  make fig-20

Run directly on an already allocated GPU:
  make fig-17 EXECUTOR=local WORKERS=1

Run all accuracy/precision experiments:
  make reproduce EXECUTOR=slurm

Create the portable archive:
  make archive

Configuration:
  cp config/local.env.example config/local.env
  Edit MODEL_ROOT, EXECUTOR, WORKERS, and optionally TABLE5_CHECKPOINT_SOURCE.
"""


if __name__ == "__main__":
    print(HELP)
