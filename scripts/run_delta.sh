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

# Define the list of alpha parameters
factors=(100.0 0.03 0.1 0.3 1.0 )

# Loop through all alpha values
for d in "${factors[@]}"; do
    echo "Running with Agent: $AGENT_NAME, Env: $ENV_NAME, Alpha: $alpha, ExpName: $EXP_NAME"
    python main.py \
        --agent "agents/$AGENT_NAME.py" \
        --env_name $2 \
        --seed "$RANDOM" \
        --delta "$d" $3
done