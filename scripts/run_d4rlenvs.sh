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
env_names=("pen-expert-v1" "door-expert-v1" "hammer-expert-v1" "relocate-expert-v1")

# Loop through all alpha values
for env_name in "${env_names[@]}"; do
    echo "Running with Agent: $AGENT_NAME, Env: $env_name, ExpName: $EXP_NAME"
    python main.py \
        --agent "agents/$AGENT_NAME.py" \
        --env_name "$env_name" \
        --exp_name "$EXP_NAME" \
        --seed "$RANDOM"
done