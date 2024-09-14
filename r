#!/bin/bash

# Default values
DEFAULT_MODEL="3.5-mini"
DEFAULT_INPUT="demo.cpp"

# Use the first argument as the model if provided, otherwise use the default
MODEL=${1:-$DEFAULT_MODEL}

# Use the second argument as the input file if provided, otherwise use the default
INPUT=${2:-$DEFAULT_INPUT}

# Remove the first two arguments if they were provided
if [ $# -gt 0 ]; then
    shift
fi
if [ $# -gt 0 ]; then
    shift
fi

# Run the command with the selected model, input file, and any additional arguments
python3 main.py "$INPUT" -m "$MODEL" "$@"
