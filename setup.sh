#!/bin/bash

python3 -m venv .venv

source .venv/bin/activate

pip install vllm

apt update

apt install neovim btop