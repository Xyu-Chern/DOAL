#!/bin/bash

# This script runs a Python script with different 'alpha' values
# while allowing the agent, environment, and an optional experiment name
# to be passed as arguments.

# Check for the correct number of arguments
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <agent_name>  [exp_name]"
    echo "Example: $0 my_new_agent my_experiment"
    exit 1
fi

# Assign command-line arguments to variables for clarity
AGENT_NAME=$1

# Check if the third argument exists and assign it
EXP_NAME=""
if [ "$#" -eq 2 ]; then
    EXP_NAME=$2
fi

# Define the list of alpha parameters
env_names=("antmaze-large-navigate-singletask-v0"   'humanoidmaze-medium-navigate-singletask-v0'  "antsoccer-arena-navigate-singletask-v0" "cube-single-play-singletask-v0"   "scene-play-singletask-v0" 'antmaze-giant-navigate-singletask-v0' 'humanoidmaze-large-navigate-singletask-v0'  "cube-double-play-singletask-v0"  "puzzle-3x3-play-singletask-v0" "puzzle-4x4-play-singletask-v0")

# Loop through all alpha values
for env_name in "${env_names[@]}"; do
    echo "Running with Agent: $AGENT_NAME, Env: $env_name, ExpName: $EXP_NAME"
    python main.py \
        --agent_name "$AGENT_NAME" \
        --env_name "$env_name" \
        --exp_name "$EXP_NAME" \
        --seed "$RANDOM"
done