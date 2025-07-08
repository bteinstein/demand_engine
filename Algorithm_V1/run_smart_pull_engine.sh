#!/bin/bash
set -e  # Exit immediately if any command fails

# Navigate to script directory and activate environment
cd "$(dirname "$0")"
# source /home/azureuser/miniconda3/envs/env_dev/bin/activate

# Set variables for better maintainability
PYTHON_EXEC="/home/azureuser/miniconda3/envs/env_dev/bin/python"
SCRIPT_NAME="run_smart_pull_engine.py"

# Execute with proper logging
echo "Starting execution: $(date)"
$PYTHON_EXEC $SCRIPT_NAME

# Clean up
# deactivate
echo "Completed successfully: $(date)"
