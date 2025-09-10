#!/bin/bash

# This script runs a Python script with different 'alpha' values
# while allowing the environment, a specific alpha value, and an optional experiment name
# to be passed as arguments.
# ./run_alpha_hd.sh antmaze-large-navigate-singletask-v0 100     
# Check for the correct number of arguments
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <env_name> <alpha_value> [exp_name]"
    echo "Example: $0 my_environment-v1 0.5 my_experiment"
    exit 1
fi

# Assign command-line arguments to variables for clarity
ENV_NAME=$1
Alpha=$2

# Assign the third argument if it exists, otherwise it will be empty
EXP_NAME=$3

# Define the list of agent names to loop through
# Corrected array declaration: no spaces around the '=' sign
agent_names=("dtrigflow" "diql" "difql")

# Loop through all agent names
for AGENT_NAME in "${agent_names[@]}"; do
    echo "Running with Agent: $AGENT_NAME, Env: $ENV_NAME, Alpha: $Alpha, ExpName: $EXP_NAME"
    python main.py \
        --agent_name "$AGENT_NAME" \
        --env_name "$ENV_NAME" \
        --alpha "$Alpha" \
        --exp_name "$EXP_NAME" \
        --seed "$RANDOM" \
        --solver diag_hess
done