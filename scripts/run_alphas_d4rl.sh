#!/bin/bash

# This script runs a Python script with different 'alpha' values
# while allowing the agent, environment, and an optional experiment name
# to be passed as arguments.

# Check for the correct number of arguments
if [ "$#" -lt 2 ] ; then
    echo "Usage: $0 <agent_name> <seed>"
    echo "Example: $0 my_new_agent seed"
    exit 1
fi

# Assign command-line arguments to variables for clarity
AGENT_NAME=$1
# Check if a seed is provided as the second argument
if [ -n "$2" ] ; then
    seed=$2
    # Shift arguments to handle the rest of the optional parameters
    shift 2
else
    seed=$RANDOM
    shift 1
fi

# pen-expert-v1		
# door-expert-v1		
# hammer-expert-v1		
# relocate-expert-v1	

# Define the list of alpha parameters

env_names=("pen-human-v1" "pen-cloned-v1" "pen-expert-v1"  "door-expert-v1"  "hammer-expert-v1" "relocate-expert-v1"  )

# Loop through all environments and alpha values
alphas=( 0.003 0.01 0.03)
for env_name in "${env_names[@]}"; do
    # Loop through all alpha values
    for alpha in "${alphas[@]}"; do
        echo "Running with Agent: $AGENT_NAME, Env: $ENV_NAME, Alpha: $alpha, ExpName: "
        python main.py \
            --agent "agents/$AGENT_NAME.py" \
            --env_name "$env_name" \
            --alpha "$alpha" \
            --run_group alpha_d4rl \
            --noretest \
            --exp_name alpha_tune $3 $4 $5 $6 $7 \
            --seed "$seed" --offline_steps 500000

    done
done
