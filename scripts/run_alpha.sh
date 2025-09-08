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
ENV_NAME=$2

# Check if the third argument exists and assign it
EXP_NAME=""
if [ "$#" -eq 3 ]; then
    EXP_NAME=$3
fi

# Define the list of alpha parameters
alphas=(  10.0 30 100 300 1000.0 3000)

# Loop through all alpha values
for alpha in "${alphas[@]}"; do
    echo "Running with Agent: $AGENT_NAME, Env: $ENV_NAME, Alpha: $alpha, ExpName: $EXP_NAME"
    python main.py \
        --agent_name "$AGENT_NAME" \
        --env_name "$ENV_NAME" \
        --alpha "$alpha" \
        --exp_name "$EXP_NAME"
done