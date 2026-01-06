#!/bin/bash
# Setup script for cross_sucking project

set -e

echo "Setting up Cross-Sucking Detection Pipeline"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo " Python 3.10+ required, found $python_version"
    exit 1
fi
echo " Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo " Installing virtualenv..."
    pip install virtualenv
    echo " Creating virtual environment..."
    virtualenv venv
fi

# Activate
echo " Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo " Upgrading pip..."
pip install --upgrade pip

# Install package
echo " Creating paths.yaml from example..."
pip install -e ".[all]"

# Create paths config from example
if [ ! -f "configs/paths.yaml" ]; then
    echo  Creating paths.yaml from example..."
    cp configs/paths.yaml.example configs/paths.yaml
    echo " Please edit configs/paths.yaml with your data paths"
fi

# Verify installation
echo ""
echo " Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "from src.cli.make_manifests import app; print('CLI modules: OK')"

echo ""
echo " Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit configs/paths.yaml with your data paths"
echo "  2. Run: cs-manifest build"
echo "  3. Run: cs-clean data/annotations/interactions.xlsx"
