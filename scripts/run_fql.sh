#!/bin/bash

# This script runs a Python script with different 'alpha' values
# while allowing the agent, environment, and an optional experiment name
# to be passed as arguments.

# Check for the correct number of arguments
if [ "$#" -lt 1 ] ; then
    echo "Usage: $0 <agent_name> [exp_name1] [exp_name2] ..."
    echo "Example: $0 my_new_agent my_experiment"
    exit 1
fi

# Assign command-line arguments to variables for clarity
AGENT_NAME=$1

# Define the list of environment names

env_names=("antmaze-large-navigate-singletask-v0" 'humanoidmaze-medium-navigate-singletask-v0' "antsoccer-arena-navigate-singletask-v0" "cube-single-play-singletask-v0" "scene-play-singletask-v0"  'humanoidmaze-large-navigate-singletask-v0' "cube-double-play-singletask-v0" "puzzle-3x3-play-singletask-v0" "puzzle-4x4-play-singletask-v0")
# Loop through all environments
for env_name in "${env_names[@]}"; do
    
    # Generate random seed
    RANDOM_SEED=$RANDOM
    
    echo "Running with Agent: FQL, Env: $env_name to Log Stats"
    
    python main.py \
        --agent "agents/fql.py" \
        --env_name "$env_name" \
        --agent.solver linear
        --seed 1 \
        --exp_name "log_stats"
done