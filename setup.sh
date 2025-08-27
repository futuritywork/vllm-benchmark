#!/bin/bash

python3 -m venv .venv

source .venv/bin/activate

pip install -r pyproject.toml

apt update
apt install neovim btop

echo "Don't forget to source the virtual environment:"
echo "source .venv/bin/activate"