#!/usr/bin/env bash
set -euo pipefail
mkdir -p training/outputs
find training/outputs -type f ! -name .gitkeep -delete
