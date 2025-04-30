#!/bin/bash

# Check if a Python script name is provided as an argument
if [ -z "$1" ]; then
    echo "Error: No Python script name provided."
    echo "Usage: ./script_name.sh <python_script_name>"
    exit 1
fi

# Get the Python script name from the first argument
SCRIPT_NAME=$1


# Find the process ID (PID) of the running script and kill it
PID=$(ps aux | grep "python $SCRIPT_NAME" | grep -v 'grep' | awk '{print $2}')

# If a PID is found, kill the process
if [ -n "$PID" ]; then
    kill -9 $PID
else
    echo "No running process found for $SCRIPT_NAME."
fi

# Run the Python script
python $SCRIPT_NAME & 
