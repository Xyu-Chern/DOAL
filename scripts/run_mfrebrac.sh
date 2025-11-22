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
SEED=$2
alpha_critic=$3


# Define the list of environment names
# env_names=("antmaze-large-navigate-singletask-v0" 'humanoidmaze-medium-navigate-singletask-v0' "antsoccer-arena-navigate-singletask-v0" "cube-single-play-singletask-v0" "scene-play-singletask-v0"  'humanoidmaze-large-navigate-singletask-v0' "cube-double-play-singletask-v0" "puzzle-3x3-play-singletask-v0" "puzzle-4x4-play-singletask-v0" )

env_names=( 'humanoidmaze-medium-navigate-singletask-v0' "antsoccer-arena-navigate-singletask-v0"  "puzzle-3x3-play-singletask-v0"  )

for env_name in "${env_names[@]}"; do
    echo "Running with Agent: $AGENT_NAME, Env: $ENV_NAME, Alpha: $alpha, ExpName: "
    python main.py \
        --agent "agents/$AGENT_NAME.py" \
        --env_name "$env_name" \
        --alpha_critic "$alpha_critic" \
        --run_group og_alpha_critic \
        --noretest $3 $4 $5 $6 $7 \
        --seed "$SEED" --offline_steps 1000000 
done
