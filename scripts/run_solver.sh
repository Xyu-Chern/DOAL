#!/bin/bash

# This script runs a Python script with different 'alpha' values
# while allowing the agent, environment, and an optional experiment name
# to be passed as arguments.

# Check for the correct number of arguments
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <agent_name> <env_name> [exp_name]"
    echo "Example: $0 my_new_agent my_new_environment-v1 my_experiment"
    exit 1
fi

# Assign command-line arguments to variables for clarity
AGENT_NAME=$1
SOLVER=$2

EXP_NAME=""
if [ "$#" -eq 3 ]; then
    EXP_NAME=$3
fi


# Define the list of alpha parameters

env_names=("antmaze-large-navigate-singletask-v0"   'humanoidmaze-medium-navigate-singletask-v0'  "antsoccer-arena-navigate-singletask-task4-v0" "cube-single-play-singletask-task2-v0"   "scene-play-singletask-task2-v0" )

# Loop through all alpha values
for env_name in "${env_names[@]}"; do
    echo "Running with Agent: $AGENT_NAME, Env: $env_name, Solver: $Solvers, ExpName: $EXP_NAME"
    python main.py \
        --agent "agents/$AGENT_NAME.py" \
        --env_name "$env_name" \
        --solver "$SOLVER" \
        --exp_name "$EXP_NAME" \
        --seed "$RANDOM"
done