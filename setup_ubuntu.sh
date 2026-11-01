#!/usr/bin/env bash
# CSIA - Ubuntu Setup Script
# Installs all required system dependencies and Python packages on Ubuntu.

set -e

echo "=== Updating package lists ==="
sudo apt-get update

echo "=== Installing system dependencies ==="
sudo apt-get install -y python3 python3-pip python3-venv python3-tk

echo "=== Installing Python dependencies ==="
pip3 install -r "$(dirname "$0")/requirements.txt" --break-system-packages 2>/dev/null || pip3 install -r "$(dirname "$0")/requirements.txt"

echo "=== Setup complete! All scripts are ready to run on Ubuntu. ==="
