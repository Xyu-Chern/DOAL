#!/bin/bash

# This script runs a Python script with different 'alpha' values
# while allowing the agent, environment, and an optional experiment name
# to be passed as arguments.

# Check for the correct number of arguments
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <agent_name> <alpha> [exp_name]"
    echo "Example: $0 my_new_agent my_new_environment-v1 my_experiment alpha"
    exit 1
fi



# Assign command-line arguments to variables for clarity
ENV_NAME=$1
Alpha=$2

# Check if the third argument exists and assign it
EXP_NAME=$3
if [ "$#" -eq 3 ]; then
    EXP_NAME=$3
fi

# Define the list of alpha parameters
)
agent_names = ( "dtrigflow" "diql" "difql")
# Loop through all alpha values
for AGENT_NAME in "${agent_names[@]}"; do
    echo "Running with Agent: $AGENT_NAME, Env: $ENV_NAME, Alpha: $Alpha, ExpName: $EXP_NAME"
    python main.py \
        --agent_name "$AGENT_NAME" \
        --env_name "$ENV_NAME" \
        --alpha "$Alpha" \
        --exp_name "$EXP_NAME" \
        --seed "$RANDOM"
done