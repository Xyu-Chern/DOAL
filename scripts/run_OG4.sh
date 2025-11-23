#!/bin/bash

# This script runs a Python script with different 'alpha' values
# while allowing the agent, a seed, and an optional experiment name
# to be passed as arguments.

# Check for the correct number of arguments
if [ "$#" -lt 1 ] ; then
    echo "Usage: $0 <agent_name> [seed] [additional_args...]"
    echo "Example: $0 my_new_agent 12345 my_experiment"
    exit 1
fi

# Assign command-line arguments to variables for clarity
AGENT_NAME=$1



# Define the list of environment names
env_names=("antmaze-large-navigate-singletask-v0" 'humanoidmaze-medium-navigate-singletask-v0' "antsoccer-arena-navigate-singletask-v0" "cube-single-play-singletask-v0" "scene-play-singletask-v0"  'humanoidmaze-large-navigate-singletask-v0' "cube-double-play-singletask-v0" "puzzle-3x3-play-singletask-v0" "puzzle-4x4-play-singletask-v0" )

seeds=(777 888)

# Loop through all environment names
for seed in "${seeds[@]}"; do
    for env_name in "${env_names[@]}"; do
        echo "Running with Agent: $AGENT_NAME, Env: $env_name, Seed: $seed, Additional Args: $@"
        python main.py \
            --agent "agents/$AGENT_NAME.py" \
            --env_name "$env_name" \
            --run_group submit_OG_rebrac \
                --noretest \
            --seed "$seed" \
            "$@"
    done
done