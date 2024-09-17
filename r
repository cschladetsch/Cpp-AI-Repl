#!/bin/bash

# Default values
DEFAULT_CODEBERT="microsoft/codebert-base"
DEFAULT_PHI="microsoft/phi-3.5-mini-instruct"
DEFAULT_INPUT="demo.cpp"
DEFAULT_TIMEOUT=300

# Initialize variables
CODEBERT_MODEL=$DEFAULT_CODEBERT
PHI_MODEL=$DEFAULT_PHI
INPUT_FILE=$DEFAULT_INPUT
TIMEOUT=$DEFAULT_TIMEOUT
DEBUG=""
LOG_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --codebert)
            CODEBERT_MODEL="$2"
            shift 2
            ;;
        --phi)
            PHI_MODEL="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        --log-file)
            LOG_FILE="--log-file $2"
            shift 2
            ;;
        *)
            INPUT_FILE="$1"
            shift
            ;;
    esac
done

# Run the Python command with the specified models, input file, and any additional arguments
echo "Running: " python3 main.py "$INPUT_FILE" --codebert "$CODEBERT_MODEL" --phi "$PHI_MODEL" --timeout "$TIMEOUT" $DEBUG $LOG_FILE
python3 main.py "$INPUT_FILE" --codebert "$CODEBERT_MODEL" --phi "$PHI_MODEL" --timeout "$TIMEOUT" $DEBUG $LOG_FILE
