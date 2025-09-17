#!/bin/bash

# This script runs a Python script with different 'alpha' values
# while allowing the agent, environment, and an optional experiment name
# to be passed as arguments.

# Check for the correct number of arguments
if [ "$#" -lt 1 ] ; then
    echo "Usage: $0 <agent_name> [1] "
    echo "Example: $0 my_new_agent my_experiment"
    exit 1
fi

# Assign command-line arguments to variables for clarity
AGENT_NAME=$1


# Define the list of alpha parameters

env_names=( 'antmaze-giant-navigate-singletask-v0' 'humanoidmaze-large-navigate-singletask-v0'  "cube-double-play-singletask-v0"  "puzzle-3x3-play-singletask-v0" "puzzle-4x4-play-singletask-v0")
# Loop through all alpha values
for env_name in "${env_names[@]}"; do
    echo "Running with Agent: $AGENT_NAME, Env: $env_name, ExpName: $2 $3 $4 $5 $6"
    python main.py \
        --agent "agents/$AGENT_NAME.py" \
        --env_name "$env_name" \
        --exp_name $2 $3 $4 $5 $6\
        --seed "$RANDOM"
done