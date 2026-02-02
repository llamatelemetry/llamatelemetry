#!/bin/bash
# Set Python 3.11 as default python3 and python for this project

echo "Setting Python 3.11 as default for llamatelemetry project..."

# Create aliases for current shell session
alias python=python3.11
alias python3=python3.11
alias pip=python3.11 -m pip
alias pip3=python3.11 -m pip

# Export for subprocesses
export PYTHON=python3.11

echo "✓ Python 3.11 set as default for current session"
echo ""
echo "To make permanent, add these lines to your ~/.bashrc:"
echo "  alias python=python3.11"
echo "  alias python3=python3.11"
echo "  alias pip=python3.11 -m pip"
echo "  alias pip3=python3.11 -m pip"
echo ""
echo "Current Python version:"
python3.11 --version
echo ""
echo "To use in this session, run:"
echo "  source $(pwd)/scripts/set_python311_default.sh"
