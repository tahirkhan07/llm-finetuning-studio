#!/usr/bin/env bash

set -e

echo "Setting up virtual environment for LLM Fine-Tuning Studio..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created at ./venv"
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Setup complete! To activate the environment, run:"
echo "source venv/bin/activate"
