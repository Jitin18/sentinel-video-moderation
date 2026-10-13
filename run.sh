#!/bin/bash
# One-command start: creates venv on first run, then launches backend + UI on http://127.0.0.1:9000
set -e
cd "$(dirname "$0")"
if [ ! -f .env ]; then echo "Missing .env — copy .env.example to .env and add your GROQ_API_KEY"; exit 1; fi
if [ ! -d .venv ]; then
  uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
fi
exec .venv/bin/python backend.py
